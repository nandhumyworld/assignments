import streamlit as st
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import openai_util as llm
import json

class QualityLevel(Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"

class ActionType(Enum):
    RETRIEVE_AGAIN = "RETRIEVE_AGAIN"
    PROCEED_WITH_ANSWER = "PROCEED_WITH_ANSWER"

class ConfidenceLevel(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class EvaluationResult:
    relevance_score: float
    completeness_score: float
    accuracy_score: float
    specificity_score: float
    overall_quality: QualityLevel
    reasoning: str

@dataclass
class CorrectionDecision:
    action: ActionType
    confidence: ConfidenceLevel
    new_query: str = None
    reasoning: str = None


class CorrectiveRAG:
    def __init__(self, session_state):
        """
        Initialize the Corrective RAG system with session state.
        
        Args:
            session_state: Streamlit session state object for maintaining state
        """
        self.session_state = session_state
        
        # Initialize session state variables if they don't exist
        if 'correction_iterations' not in self.session_state:
            self.session_state.correction_iterations = 0
        if 'correction_history' not in self.session_state:
            self.session_state.correction_history = []
    
    def context_evaluation(self, user_query: str, retrieved_context: str, session_state) -> EvaluationResult:
        """
        Evaluate the quality of retrieved context against the user query using OpenAI LLM.
        
        Args:
            user_query (str): The original user query
            retrieved_context (str): The context retrieved from the knowledge base
            session_state: Streamlit session state object
            
        Returns:
            EvaluationResult: Detailed evaluation scores and overall quality from LLM
        """
        # Store current evaluation in session state
        session_state.current_query = user_query
        session_state.current_context = retrieved_context
        
        # Construct evaluation prompt
        evaluation_prompt = f"""
        Rate the following retrieved context for the given query:

        Query: {user_query} 
        Retrieved Context: {retrieved_context} 

        Evaluation Criteria: 
        1. Relevance Score (0-1): How well does the context address the query? 
        2. Completeness Score (0-1): Does the context provide sufficient information? 
        3. Accuracy Score (0-1): Is the information factually correct? 
        4. Specificity Score (0-1): Is the context specific enough for the query? 

        Overall Quality: [EXCELLENT/GOOD/FAIR/POOR]

        Respond with ONLY this JSON format:
        {{
            "relevance_score": 0.8,
            "completeness_score": 0.7,
            "accuracy_score": 0.9,
            "specificity_score": 0.6,
            "overall_quality": "GOOD",
            "reasoning": "Detailed explanation of the evaluation"
        }}
        """
        
        try:
            client = llm.get_OpenAI_client()
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a Corrective RAG system that evaluates retrieved context quality and corrects retrieval when necessary. Always respond with valid JSON only."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            # Parse JSON response
            llm_response = response.choices[0].message.content.strip()
            st.markdown("**LLM Evaluation Response:**")
            st.code(llm_response, language='json')

            evaluation_data = json.loads(llm_response)
            
            # Create EvaluationResult object
            evaluation_result = EvaluationResult(
                relevance_score=float(evaluation_data["relevance_score"]),
                completeness_score=float(evaluation_data["completeness_score"]),
                accuracy_score=float(evaluation_data["accuracy_score"]),
                specificity_score=float(evaluation_data["specificity_score"]),
                overall_quality=QualityLevel(evaluation_data["overall_quality"]),
                reasoning=evaluation_data["reasoning"]
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to heuristic evaluation if LLM fails
            st.warning(f"LLM evaluation failed ({str(e)}), using fallback heuristics")
            evaluation_result = self._fallback_context_evaluation(user_query, retrieved_context)
        
        # Store in session state
        session_state.last_evaluation = evaluation_result
        
        return evaluation_result
    
    def correction_decision(self, evaluation_result: EvaluationResult, user_query: str) -> CorrectionDecision:
        """
        Make a decision on whether to retrieve again or proceed using OpenAI LLM.
        
        Args:
            evaluation_result (EvaluationResult): Result from context evaluation
            user_query (str): Original user query
            
        Returns:
            CorrectionDecision: Decision on next action with reasoning from LLM
        """
        # Construct correction decision prompt
        correction_prompt = f"""
        You are an expert RAG system decision maker. Based on the context evaluation results, decide the next action.

        Original Query: {user_query}
        Evaluation Results:
        - Relevance Score: {evaluation_result.relevance_score}
        - Completeness Score: {evaluation_result.completeness_score}
        - Accuracy Score: {evaluation_result.accuracy_score}
        - Specificity Score: {evaluation_result.specificity_score}
        - Overall Quality: {evaluation_result.overall_quality.value}
        - Reasoning: {evaluation_result.reasoning}

        Respond with ONLY this JSON format:
        {{
            "confidence": "HIGH" or "MEDIUM" or "LOW",
            "new_query": "refined query if action is RETRIEVE_AGAIN, otherwise null",
            "reasoning": "detailed explanation for the decision"
        }}
        """
        
        try:
            client = llm.get_OpenAI_client()
            # Call OpenAI API
            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                        {"role": "system", "content": "You are an expert RAG decision maker. Always respond with valid JSON only."},
                        {"role": "user", "content": correction_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
            # Parse JSON response
            llm_response = response.choices[0].message.content.strip()
            st.markdown("**LLM Correction Decision Response:**")
            st.code(llm_response, language='json')
            decision_data = json.loads(llm_response)

             
            if evaluation_result.overall_quality.value.strip() == QualityLevel.POOR.value or evaluation_result.overall_quality.value.strip() == QualityLevel.FAIR.value:
                # Create CorrectionDecision object
                decision = CorrectionDecision(
                    action=ActionType.RETRIEVE_AGAIN,
                    confidence=ConfidenceLevel(decision_data["confidence"]),
                    new_query=decision_data.get("new_query"),
                    reasoning=decision_data["reasoning"]
                )
            elif evaluation_result.overall_quality.value.strip() == QualityLevel.GOOD.value or evaluation_result.overall_quality.value.strip() == QualityLevel.EXCELLENT.value:
                # Create CorrectionDecision object
                decision = CorrectionDecision(
                    action=ActionType.PROCEED_WITH_ANSWER,
                    confidence=ConfidenceLevel(decision_data["confidence"]),
                    new_query=decision_data.get("new_query"),
                    reasoning=decision_data["reasoning"]
                )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to heuristic decision if LLM fails
            st.warning(f"LLM decision failed ({str(e)}), using fallback heuristics")
            decision = self._fallback_correction_decision(evaluation_result, user_query)
        
        # Update session state
        # self.session_state.correction_iterations += 1
        # self.session_state.correction_history.append({
        #     'iteration': self.session_state.correction_iterations,
        #     'evaluation': evaluation_result,
        #     'decision': decision,
        #     'timestamp': st.session_state.get('current_time', 'unknown')
        # })
        
        return decision
    
    def format_response(self, user_query: str, retrieved_context: str, evaluation_result: EvaluationResult, 
                       decision: CorrectionDecision, answer: str = None, sources: List[str] = None) -> str:
        """
        Format the final response according to the specified format.
        
        Args:
            user_query (str): Original user query
            retrieved_context (str): Retrieved context
            evaluation_result (EvaluationResult): Evaluation results
            decision (CorrectionDecision): Correction decision
            answer (str): Generated answer (if proceeding)
            sources (List[str]): Context sources
            
        Returns:
            str: Formatted response string
        """
        quality_level = evaluation_result.overall_quality.value
        confidence_level = decision.confidence.value

        if decision.action.value.strip() == ActionType.PROCEED_WITH_ANSWER.value.strip():
            response_text = answer if answer else "Generated response based on retrieved context"
            sources_text = ", ".join(sources) if sources else "Retrieved context sources"
        else:
            response_text = f"Context quality insufficient. Refining query: {decision.new_query}"
            sources_text = "Re-retrieval required"
        
        formatted_response = f"""
            🔍 Context Quality: {quality_level}
            📊 Confidence Level: {confidence_level}
            🎯 Answer: {response_text}
            📚 Sources: {sources_text}
                    """.strip()
        
        return formatted_response
    
    def _determine_overall_quality(self, average_score: float) -> QualityLevel:
        """Determine overall quality based on average score."""
        if average_score >= 0.8:
            return QualityLevel.EXCELLENT
        elif average_score >= 0.6:
            return QualityLevel.GOOD
        elif average_score >= 0.4:
            return QualityLevel.FAIR
        else:
            return QualityLevel.POOR
    
    def _fallback_context_evaluation(self, user_query: str, retrieved_context: str) -> EvaluationResult:
        """
        Fallback heuristic evaluation when LLM fails.
        """
        # Calculate individual scores using heuristic methods
        relevance_score = self._calculate_relevance_score(user_query, retrieved_context)
        completeness_score = self._calculate_completeness_score(user_query, retrieved_context)
        accuracy_score = self._calculate_accuracy_score(retrieved_context)
        specificity_score = self._calculate_specificity_score(user_query, retrieved_context)
        
        # Determine overall quality
        average_score = (relevance_score + completeness_score + accuracy_score + specificity_score) / 4
        overall_quality = self._determine_overall_quality(average_score)
        
        # Generate reasoning
        reasoning = self._generate_evaluation_reasoning(
            relevance_score, completeness_score, accuracy_score, specificity_score, overall_quality
        )
        
        return EvaluationResult(
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            specificity_score=specificity_score,
            overall_quality=overall_quality,
            reasoning=reasoning
        )
    
    def _generate_evaluation_reasoning(self, relevance: float, completeness: float, 
                                     accuracy: float, specificity: float, 
                                     overall_quality: QualityLevel) -> str:
        """Generate human-readable reasoning for the evaluation."""
        reasons = []
        
        if relevance < 0.5:
            reasons.append("Low relevance to query")
        if completeness < 0.5:
            reasons.append("Incomplete information")
        if accuracy < 0.5:
            reasons.append("Questionable accuracy")
        if specificity < 0.5:
            reasons.append("Lacks specificity")
        
        if not reasons:
            return f"Context demonstrates {overall_quality.value.lower()} quality across all evaluation criteria"
        else:
            return f"Quality issues: {', '.join(reasons)}"
        
    def get_correction_history(self) -> List[Dict]:
        """Get the history of correction iterations."""
        return self.session_state.correction_history
    
    def reset_session(self):
        """Reset the correction session."""
        self.session_state.correction_iterations = 0
        self.session_state.correction_history = []
        if hasattr(self.session_state, 'current_query'):
            del self.session_state.current_query
        if hasattr(self.session_state, 'current_context'):
            del self.session_state.current_context
        if hasattr(self.session_state, 'last_evaluation'):
            del self.session_state.last_evaluation