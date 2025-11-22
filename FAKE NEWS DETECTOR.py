import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import warnings
warnings.filterwarnings('ignore')

class MinimalFakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # Use a single model for simplicity
        self.model = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'
        )
        
        self.is_trained = False
        
        # Basic fake news patterns (manually defined)
        self.fake_news_patterns = {
            "sensational_words": [
                'shocking', 'breaking', 'secret', 'miracle', 'amazing', 'unbelievable',
                'astounding', 'incredible', 'mind-blowing', 'earth-shattering'
            ],
            "absolute_claims": [
                'proven', 'confirmed', '100%', 'guaranteed', 'definitely', 'absolutely',
                'certainly', 'undeniably', 'irrefutable', 'conclusive'
            ],
            "conspiracy_terms": [
                'cover-up', 'secret', 'leaked', 'hidden', 'they don\'t want you to know',
                'suppressed', 'censored', 'mainstream media', 'establishment'
            ],
            "emotional_triggers": [
                'urgent', 'warning', 'danger', 'emergency', 'alert', 'critical',
                'disturbing', 'outrageous', 'horrifying', 'terrifying'
            ],
            "fake_authority": [
                'experts say', 'studies show', 'sources confirm', 'doctors reveal',
                'scientists prove', 'research indicates'
            ]
        }
        
        # Basic English stopwords (common words to remove)
        self.basic_stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
            'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
            'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
            'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
            'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
            'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
            'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', 
            "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
            'wouldn', "wouldn't"
        }
    
    def extract_basic_features(self, text):
        """Extract basic linguistic features without NLTK"""
        features = {}
        text_lower = text.lower()
        
        # Basic text statistics using simple split
        words = text.split()
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['all_caps_count'] = sum(1 for word in words if word.isupper() and len(word) > 1)
        
        # Pattern matching for fake news indicators
        for pattern_type, words_list in self.fake_news_patterns.items():
            features[f'{pattern_type}_count'] = sum(1 for word in words_list if word in text_lower)
        
        # Clickbait score
        features['clickbait_score'] = self.calculate_clickbait_score(text)
        
        # Sentiment-like features (basic)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'horrible', 'awful', 'disastrous', 'shocking']
        features['positive_word_count'] = sum(1 for word in positive_words if word in text_lower)
        features['negative_word_count'] = sum(1 for word in negative_words if word in text_lower)
        
        return features
    
    def calculate_clickbait_score(self, text):
        """Calculate clickbait likelihood score"""
        score = 0
        text_lower = text.lower()
        
        # Common clickbait phrases
        clickbait_phrases = [
            'you won\'t believe', 'what happened next', 'this will shock you',
            'the truth about', 'they don\'t want you to know', 'going viral',
            'will blow your mind', 'secret revealed'
        ]
        
        for phrase in clickbait_phrases:
            if phrase in text_lower:
                score += 2
        
        # Excessive punctuation
        if text.count('!') > 2:
            score += 1
        if text.count('?') > 3:
            score += 1
            
        # All caps words
        if any(word.isupper() and len(word) > 3 for word in text.split()):
            score += 2
            
        return min(score, 10)  # Normalize to 0-10 scale
    
    def preprocess_text(self, text):
        """Basic text preprocessing without NLTK"""
        if isinstance(text, float):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords using basic list
        words = [word for word in text.split() if word not in self.basic_stopwords]
        
        return ' '.join(words)
    
    def prepare_data(self, texts, labels=None):
        """Prepare data with basic features"""
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        if labels is not None:
            # TF-IDF features
            X_tfidf = self.vectorizer.fit_transform(processed_texts)
            
            # Basic linguistic features
            linguistic_features = []
            for text in texts:
                features = self.extract_basic_features(text)
                linguistic_features.append(list(features.values()))
            
            X_linguistic = np.array(linguistic_features)
            
            # Combine features
            from scipy.sparse import hstack
            X_combined = hstack([X_tfidf, X_linguistic])
            
            return X_combined, np.array(labels)
        else:
            # For prediction
            X_tfidf = self.vectorizer.transform(processed_texts)
            
            linguistic_features = []
            for text in texts:
                features = self.extract_basic_features(text)
                linguistic_features.append(list(features.values()))
            
            X_linguistic = np.array(linguistic_features)
            
            from scipy.sparse import hstack
            X_combined = hstack([X_tfidf, X_linguistic])
            
            return X_combined
    
    def train(self, texts, labels):
        """Train the detector"""
        print("Preprocessing texts and extracting features...")
        X, y = self.prepare_data(texts, labels)
        
        print("Training model...")
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training accuracy
        train_pred = self.model.predict(X)
        train_accuracy = accuracy_score(y, train_pred)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        return train_accuracy
    
    def predict_with_confidence(self, texts):
        """Make predictions with confidence scores"""
        if not self.is_trained:
            raise Exception("Model must be trained before prediction!")
        
        X = self.prepare_data(texts)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Simple confidence calculation
        confidences = [max(prob) for prob in probabilities]
        
        return predictions, probabilities, confidences
    
    def analyze_news_characteristics(self, text):
        """Comprehensive analysis of news characteristics"""
        features = self.extract_basic_features(text)
        analysis = {
            'text': text,
            'word_count': features['word_count'],
            'clickbait_score': features['clickbait_score'],
            'fake_indicators_count': sum([
                features['sensational_words_count'],
                features['absolute_claims_count'],
                features['conspiracy_terms_count'],
                features['emotional_triggers_count'],
                features['fake_authority_count']
            ]),
            'specific_indicators': {}
        }
        
        # Detailed indicator analysis
        for pattern_type in self.fake_news_patterns.keys():
            count = features[f'{pattern_type}_count']
            if count > 0:
                analysis['specific_indicators'][pattern_type] = count
        
        return analysis

def create_comprehensive_dataset():
    """Create a comprehensive dataset for training"""
    
    # Real news examples
    real_news = [
        "Scientists discover new species in Amazon rainforest during biodiversity survey",
        "Global leaders meet at climate summit to discuss carbon emission reduction targets",
        "New study shows benefits of regular exercise for mental health",
        "Economy shows signs of recovery with GDP growth and increased job numbers",
        "Breakthrough in solar panel technology announced by researchers",
        "Local community raises funds for new school building through fundraising campaign",
        "Medical researchers develop new treatment for diabetes",
        "International Space Station completes 25 years of continuous human presence",
        "Education department announces new scholarship programs for STEM students",
        "City council approves plan for public transportation improvement",
        "Federal Reserve maintains interest rates amid stable economic indicators",
        "New research reveals Mediterranean diet associated with lower heart disease risk",
        "Company announces improvement in battery life for electric vehicles",
        "Study finds music education improves cognitive development in children",
        "Government launches initiative providing grants to small businesses",
        "Researchers make progress in Alzheimer's treatment with new drug showing promise",
        "United Nations reports progress in achieving sustainable development goals",
        "Local hospital introduces new robotic surgery system for improved patient outcomes",
        "Tech company announces partnership with universities to promote computer science education",
        "Agricultural department reports record crop yields due to improved farming techniques"
    ]
    
    # Fake news examples
    fake_news = [
        "BREAKING: Aliens landed in New York Central Park, government in emergency session",
        "Celebrity doctor reveals secret immortality potion that pharmaceutical companies are suppressing",
        "SHOCKING: Eating tomatoes causes instant aging according to leaked FDA documents",
        "Government admits using weather control satellites to manipulate hurricane paths",
        "Miracle cure discovered in remote village makes all diseases disappear overnight",
        "Famous actor confesses entire career was computer-generated in bombshell interview",
        "Secret underground cities housing millions found beneath New York and London",
        "New law will require microchip implants in all newborns starting next year",
        "Ancient Mayan prophecy predicts world ending next week, NASA scientists confirm",
        "Drinking water in major cities found to contain mind control chemicals",
        "Independent study proves vaccines contain tracking devices and alter DNA",
        "Medical experts warn eating bananas after 6 PM causes irreversible health damage",
        "New evidence reveals moon landing was completely staged in Hollywood studio",
        "Secret society of bankers controls world economy through hidden financial system",
        "New smartphone app can read your thoughts and sell data to advertisers",
        "Government planning to implement social credit system like China by next year",
        "5G towers confirmed to be causing bird deaths and health issues nationwide",
        "Leaked documents show celebrities involved in secret child trafficking ring",
        "Drinking coffee proven to cause cancer according to hidden research findings",
        "Earth heading for 15 days of complete darkness due to rare astronomical event"
    ]
    
    texts = real_news + fake_news
    labels = [0] * len(real_news) + [1] * len(fake_news)  # 0 = real, 1 = fake
    
    return pd.DataFrame({'text': texts, 'label': labels})

def get_detailed_analysis(detector, text):
    """Get comprehensive analysis of a news text"""
    prediction, probabilities, confidence = detector.predict_with_confidence([text])
    analysis = detector.analyze_news_characteristics(text)
    
    result = {
        'prediction': 'FAKE NEWS' if prediction[0] == 1 else 'REAL NEWS',
        'confidence': confidence[0],
        'probability_fake': probabilities[0][1],
        'probability_real': probabilities[0][0],
        'analysis': analysis,
        'recommendation': ''
    }
    
    # Generate recommendation
    if confidence[0] > 0.8:
        result['recommendation'] = "High confidence in prediction"
    elif confidence[0] > 0.6:
        result['recommendation'] = "Moderate confidence - consider verifying with fact-checkers"
    else:
        result['recommendation'] = "Low confidence - manual verification recommended"
    
    return result

def display_detailed_results(results):
    """Display comprehensive analysis results"""
    for i, result in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"ANALYSIS RESULT {i}")
        print(f"{'='*60}")
        
        # Prediction with simple formatting
        if result['prediction'] == 'FAKE NEWS':
            prediction_marker = "🔴"
        else:
            prediction_marker = "🟢"
        
        print(f"Text: {result['analysis']['text']}")
        print(f"Prediction: {prediction_marker} {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Fake Probability: {result['probability_fake']:.4f}")
        print(f"Real Probability: {result['probability_real']:.4f}")
        print(f"Recommendation: {result['recommendation']}")
        
        # Detailed analysis
        print(f"\nDetailed Analysis:")
        print(f"  • Word Count: {result['analysis']['word_count']}")
        print(f"  • Clickbait Score: {result['analysis']['clickbait_score']}/10")
        print(f"  • Fake News Indicators: {result['analysis']['fake_indicators_count']}")
        
        if result['analysis']['specific_indicators']:
            print(f"  • Specific Indicators Found:")
            for indicator, count in result['analysis']['specific_indicators'].items():
                print(f"    - {indicator.replace('_', ' ').title()}: {count}")

def main():
    """Main function"""
    print("=" * 60)
    print("      MINIMAL FAKE NEWS DETECTION SYSTEM")
    print("=" * 60)
    
    # Initialize detector
    print("\nInitializing Fake News Detector...")
    detector = MinimalFakeNewsDetector()
    
    # Create and prepare dataset
    print("Creating dataset...")
    df = create_comprehensive_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.25, random_state=42, stratify=df['label']
    )
    
    # Train model
    print("Training model...")
    detector.train(X_train, y_train)
    
    # Evaluate model
    print("\nModel Evaluation:")
    test_predictions, test_probabilities, test_confidences = detector.predict_with_confidence(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions, target_names=['Real News', 'Fake News']))
    
    # User interaction loop
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Analyze single news item")
        print("2. Analyze multiple news items")
        print("3. Test with example news")
        print("4. View model performance")
        print("5. Exit")
        
        try:
            choice = input("\nChoose option (1-5): ").strip()
            
            if choice == '1':
                text = input("\nEnter the news text to analyze: ").strip()
                if text:
                    result = get_detailed_analysis(detector, text)
                    display_detailed_results([result])
                else:
                    print("Please enter some text.")
            
            elif choice == '2':
                print("\nEnter multiple news texts (empty line to finish):")
                texts = []
                while True:
                    text = input("> ").strip()
                    if not text:
                        if texts:
                            break
                        else:
                            continue
                    texts.append(text)
                
                results = []
                for text in texts:
                    results.append(get_detailed_analysis(detector, text))
                display_detailed_results(results)
            
            elif choice == '3':
                # Test with examples
                examples = [
                    "Breaking: Scientists make amazing discovery that will change everything forever!",
                    "City council approves new park development plan",
                    "Secret government program controls all social media platforms according to insiders",
                    "New study shows benefits of regular exercise for cardiovascular health"
                ]
                
                print("\nTesting with example news:")
                results = []
                for example in examples:
                    results.append(get_detailed_analysis(detector, example))
                display_detailed_results(results)
            
            elif choice == '4':
                print(f"\nModel Performance Summary:")
                print(f"Training samples: {len(X_train)}")
                print(f"Testing samples: {len(X_test)}")
                print(f"Test Accuracy: {test_accuracy:.4f}")
                
                # Show confusion matrix
                cm = confusion_matrix(y_test, test_predictions)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.show()
            
            elif choice == '5':
                print("Thank you for using the Fake News Detection System!")
                break
            
            else:
                print("Please enter a valid option (1-5).")
                
        except Exception as e:
            print(f"An error occurred: {e}")

# Requirements installation helper
def check_requirements():
    """Check if required packages are installed"""
    required_packages = {
        'pandas': 'pd',
        'numpy': 'np', 
        'matplotlib': 'plt',
        'seaborn': 'sns',
        'sklearn': 'sklearn'
    }
    
    missing = []
    for package, short_name in required_packages.items():
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing packages:", missing)
        print("Install with: pip install", " ".join(missing))
        return False
    return True

if __name__ == "__main__":
    if check_requirements():
        main()
