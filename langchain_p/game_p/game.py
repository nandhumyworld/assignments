import streamlit as st
import random
import re
from typing import List, Dict, Tuple

# Word categories and puzzles
WORD_PUZZLES = {
    "animals": [
        {"word": "ELEPHANT", "clue": "Large gray mammal with a trunk"},
        {"word": "BUTTERFLY", "clue": "Colorful insect that starts as a caterpillar"},
        {"word": "PENGUIN", "clue": "Black and white bird that can't fly but swims well"},
        {"word": "GIRAFFE", "clue": "Tallest animal in the world with a long neck"},
        {"word": "DOLPHIN", "clue": "Intelligent marine mammal that clicks and whistles"}
    ],
    "fruits": [
        {"word": "PINEAPPLE", "clue": "Tropical fruit with a spiky exterior and sweet interior"},
        {"word": "STRAWBERRY", "clue": "Small red fruit with seeds on the outside"},
        {"word": "WATERMELON", "clue": "Large green fruit that's mostly water and has black seeds"},
        {"word": "BLUEBERRY", "clue": "Small round blue fruit often used in muffins"},
        {"word": "POMEGRANATE", "clue": "Red fruit filled with many small seeds"}
    ],
    "countries": [
        {"word": "AUSTRALIA", "clue": "Island continent known for kangaroos and koalas"},
        {"word": "SWITZERLAND", "clue": "Mountainous country famous for chocolate and watches"},
        {"word": "MADAGASCAR", "clue": "Large island nation off the coast of Africa"},
        {"word": "PHILIPPINES", "clue": "Southeast Asian archipelago of over 7,000 islands"},
        {"word": "BANGLADESH", "clue": "South Asian country bordered by India and Myanmar"}
    ]
}

class WordPuzzleGame:
    def __init__(self):
        self.reset_game()
    
    def reset_game(self):
        """Reset the game state"""
        category = random.choice(list(WORD_PUZZLES.keys()))
        puzzle = random.choice(WORD_PUZZLES[category])
        
        self.target_word = puzzle["word"]
        self.clue = puzzle["clue"]
        self.category = category.title()
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.max_wrong_guesses = 6
        self.game_over = False
        self.won = False
        self.hints_used = 0
        self.max_hints = 2
    
    def get_display_word(self) -> str:
        """Get the current state of the word with guessed letters revealed"""
        return ''.join([letter if letter in self.guessed_letters else '_' for letter in self.target_word])
    
    def guess_letter(self, letter: str) -> Tuple[bool, str]:
        """Process a letter guess"""
        letter = letter.upper()
        
        if letter in self.guessed_letters:
            return False, f"You already guessed '{letter}'. Try a different letter!"
        
        self.guessed_letters.add(letter)
        
        if letter in self.target_word:
            # Check if word is complete
            if set(self.target_word) <= self.guessed_letters:
                self.won = True
                self.game_over = True
                return True, f"🎉 Excellent! '{letter}' is correct! You solved it: {self.target_word}!"
            else:
                return True, f"✅ Great! '{letter}' is in the word!"
        else:
            self.wrong_guesses += 1
            if self.wrong_guesses >= self.max_wrong_guesses:
                self.game_over = True
                return False, f"❌ Sorry, '{letter}' is not in the word. Game over! The word was: {self.target_word}"
            else:
                remaining = self.max_wrong_guesses - self.wrong_guesses
                return False, f"❌ Sorry, '{letter}' is not in the word. {remaining} wrong guesses remaining."
    
    def get_hint(self) -> str:
        """Provide a hint to the player"""
        if self.hints_used >= self.max_hints:
            return "🚫 No more hints available!"
        
        # Find unguessed letters
        unguessed = [letter for letter in self.target_word if letter not in self.guessed_letters]
        
        if unguessed:
            hint_letter = random.choice(unguessed)
            self.guessed_letters.add(hint_letter)
            self.hints_used += 1
            remaining_hints = self.max_hints - self.hints_used
            
            # Check if word is complete after hint
            if set(self.target_word) <= self.guessed_letters:
                self.won = True
                self.game_over = True
                return f"💡 Hint: The letter '{hint_letter}' is in the word! You solved it: {self.target_word}!"
            
            return f"💡 Hint: The letter '{hint_letter}' is in the word! ({remaining_hints} hints remaining)"
        
        return "💡 You've already guessed all the letters!"

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'game' not in st.session_state:
        st.session_state.game = WordPuzzleGame()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'game_started' not in st.session_state:
        st.session_state.game_started = False

def add_to_chat(role: str, message: str):
    """Add message to chat history"""
    st.session_state.chat_history.append({"role": role, "message": message})

def display_game_state():
    """Display current game state"""
    game = st.session_state.game
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Category", game.category)
    with col2:
        st.metric("Wrong Guesses", f"{game.wrong_guesses}/{game.max_wrong_guesses}")
    with col3:
        st.metric("Hints Used", f"{game.hints_used}/{game.max_hints}")
    
    # Display word progress
    display_word = game.get_display_word()
    st.markdown(f"### Word: `{' '.join(display_word)}`")
    
    # Display guessed letters
    if game.guessed_letters:
        guessed_sorted = sorted(list(game.guessed_letters))
        st.markdown(f"**Guessed letters:** {', '.join(guessed_sorted)}")

def process_user_input(user_input: str) -> str:
    """Process user input and return bot response"""
    game = st.session_state.game
    user_input = user_input.strip().lower()
    
    # Check for commands
    if user_input in ['help', 'rules']:
        return """📋 **Game Rules:**
- Guess letters one at a time to reveal the hidden word
- You have 6 wrong guesses before the game ends
- You can use up to 2 hints during the game
- Type 'hint' to get a hint
- Type 'new game' to start over"""
    
    elif user_input == 'hint':
        return game.get_hint()
    
    elif user_input in ['new game', 'restart', 'reset']:
        game.reset_game()
        st.session_state.game_started = True
        return f"""🎮 **New Game Started!**
Category: {game.category}
Clue: {game.clue}

Let's play! Guess a letter to start revealing the word."""
    
    elif user_input == 'quit':
        return "👋 Thanks for playing! Type 'new game' if you want to play again."
    
    # Check if it's a single letter guess
    elif len(user_input) == 1 and user_input.isalpha():
        if game.game_over:
            return "🎮 Game is over! Type 'new game' to start a new puzzle."
        
        success, message = game.guess_letter(user_input)
        return message
    
    # Check if it's a full word guess
    elif user_input.replace(' ', '').isalpha():
        if game.game_over:
            return "🎮 Game is over! Type 'new game' to start a new puzzle."
        
        guess = user_input.replace(' ', '').upper()
        if guess == game.target_word:
            game.won = True
            game.game_over = True
            return f"🎉 Amazing! You guessed the entire word: {game.target_word}!"
        else:
            game.wrong_guesses += 1
            if game.wrong_guesses >= game.max_wrong_guesses:
                game.game_over = True
                return f"❌ Sorry, '{guess}' is not correct. Game over! The word was: {game.target_word}"
            else:
                remaining = game.max_wrong_guesses - game.wrong_guesses
                return f"❌ Sorry, '{guess}' is not the word. {remaining} wrong guesses remaining."
    
    else:
        return """🤔 I didn't understand that. You can:
- Type a single letter to guess
- Type the full word if you think you know it
- Type 'hint' for a clue
- Type 'help' for rules
- Type 'new game' to start over"""

def main():
    st.set_page_config(
        page_title="Word Puzzle Chatbot",
        page_icon="🎯",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🎯 Word Puzzle Chatbot Game")
    st.markdown("*Guess the hidden word by chatting with your AI game master!*")
    
    # Sidebar with game info
    with st.sidebar:
        st.header("🎮 Game Status")
        
        if st.session_state.game_started:
            display_game_state()
        else:
            st.info("Start a new game to begin!")
        
        st.markdown("---")
        st.markdown("""
        **Commands:**
        - `help` - Show rules
        - `hint` - Get a clue
        - `new game` - Start over
        - `quit` - End game
        """)
    
    # Chat interface
    st.header("💬 Chat with Game Master")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                with st.chat_message("user"):
                    st.write(chat["message"])
            else:
                with st.chat_message("assistant"):
                    st.write(chat["message"])
    
    # Welcome message
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            welcome_msg = """👋 Welcome to Word Puzzle Game! 

I'm your AI game master. I'll give you clues and you guess the hidden word!

Type 'new game' to start your first puzzle, or 'help' to see the rules."""
            st.write(welcome_msg)
            add_to_chat("assistant", welcome_msg)
    
    # User input
    if prompt := st.chat_input("Type your guess or command..."):
        # Add user message to chat
        with st.chat_message("user"):
            st.write(prompt)
        add_to_chat("user", prompt)
        
        # Process input and get response
        response = process_user_input(prompt)
        
        # Add bot response to chat
        with st.chat_message("assistant"):
            st.write(response)
        add_to_chat("assistant", response)
        
        # Auto-start first game
        if not st.session_state.game_started and prompt.lower() in ['new game', 'start', 'begin']:
            st.session_state.game_started = True
        
        # Rerun to update the display
        st.rerun()
    
    # Quick action buttons
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🆕 New Game"):
            game = st.session_state.game
            game.reset_game()
            st.session_state.game_started = True
            response = f"""🎮 **New Game Started!**
Category: {game.category}
Clue: {game.clue}

Let's play! Guess a letter to start revealing the word."""
            add_to_chat("assistant", response)
            st.rerun()
    
    with col2:
        if st.button("💡 Hint"):
            if st.session_state.game_started:
                response = st.session_state.game.get_hint()
                add_to_chat("user", "hint")
                add_to_chat("assistant", response)
                st.rerun()
    
    with col3:
        if st.button("📋 Help"):
            response = process_user_input("help")
            add_to_chat("user", "help")
            add_to_chat("assistant", response)
            st.rerun()
    
    with col4:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()