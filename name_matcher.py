import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
from collections import defaultdict
import re

# ============================
# Option 1: Using FuzzyWuzzy (Easy to implement)
# ============================
from fuzzywuzzy import fuzz, process
from fuzzywuzzy.utils import full_process

class FuzzyNameMatcher:
    def __init__(self, names_dataset: List[str]):
        """
        Initialize with a dataset of names
        """
        self.names_dataset = names_dataset
        
    def find_best_match(self, input_name: str) -> Tuple[str, int]:
        """
        Find the single best matching name
        """
        # Get the best match with score
        best_match, score = process.extractOne(
            input_name, 
            self.names_dataset,
            scorer=fuzz.token_sort_ratio
        )
        return best_match, score
    
    def find_ranked_matches(self, input_name: str, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Find ranked list of similar names
        """
        matches = process.extract(
            input_name,
            self.names_dataset,
            scorer=fuzz.token_sort_ratio,
            limit=limit
        )
        return matches
    
    def find_matches_with_threshold(self, input_name: str, threshold: int = 70) -> List[Tuple[str, int]]:
        """
        Find all matches above a certain similarity threshold
        """
        all_matches = []
        for name in self.names_dataset:
            score = fuzz.token_sort_ratio(input_name, name)
            if score >= threshold:
                all_matches.append((name, score))
        
        # Sort by score in descending order
        return sorted(all_matches, key=lambda x: x[1], reverse=True)

# ============================
# Option 2: Using Vector Embeddings (More Advanced)
# ============================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class VectorNameMatcher:
    def __init__(self, names_dataset: List[str]):
        """
        Initialize with TF-IDF vectorizer
        """
        self.names_dataset = names_dataset
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
        self.name_vectors = self.vectorizer.fit_transform(names_dataset)
        
    def find_best_match(self, input_name: str) -> Tuple[str, float]:
        """
        Find best match using cosine similarity
        """
        # Transform input name to vector
        input_vector = self.vectorizer.transform([input_name])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(input_vector, self.name_vectors).flatten()
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx] * 100  # Convert to percentage
        
        return self.names_dataset[best_idx], round(best_score, 2)
    
    def find_ranked_matches(self, input_name: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Find ranked list of similar names
        """
        # Transform input name to vector
        input_vector = self.vectorizer.transform([input_name])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(input_vector, self.name_vectors).flatten()
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:limit]
        
        matches = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include positive similarities
                score = similarities[idx] * 100  # Convert to percentage
                matches.append((self.names_dataset[idx], round(score, 2)))
        
        return matches

# ============================
# Option 3: Hybrid Approach (Combines multiple methods)
# ============================
class HybridNameMatcher:
    def __init__(self, names_dataset: List[str]):
        self.names_dataset = names_dataset
        self.fuzzy_matcher = FuzzyNameMatcher(names_dataset)
        self.vector_matcher = VectorNameMatcher(names_dataset)
        
    def find_best_match(self, input_name: str) -> Dict:
        """
        Find best match using hybrid approach
        """
        # Get results from both methods
        fuzzy_best, fuzzy_score = self.fuzzy_matcher.find_best_match(input_name)
        vector_best, vector_score = self.vector_matcher.find_best_match(input_name)
        
        # Weighted average (can adjust weights based on testing)
        avg_score = (fuzzy_score * 0.6 + vector_score * 0.4)
        
        # Decide which result to use (or combine)
        if fuzzy_score >= vector_score:
            best_match = fuzzy_best
            final_score = fuzzy_score
            method = "fuzzy"
        else:
            best_match = vector_best
            final_score = vector_score
            method = "vector"
        
        return {
            "best_match": best_match,
            "similarity_score": round(final_score, 2),
            "method_used": method,
            "fuzzy_match": (fuzzy_best, fuzzy_score),
            "vector_match": (vector_best, vector_score)
        }
    
    def find_ranked_matches(self, input_name: str, limit: int = 10) -> List[Dict]:
        """
        Find ranked matches using combined scores
        """
        # Get matches from both methods
        fuzzy_matches = dict(self.fuzzy_matcher.find_ranked_matches(input_name, limit*2))
        vector_matches = dict(self.vector_matcher.find_ranked_matches(input_name, limit*2))
        
        # Combine scores
        combined_scores = defaultdict(float)
        all_names = set(list(fuzzy_matches.keys()) + list(vector_matches.keys()))
        
        for name in all_names:
            fuzzy_score = fuzzy_matches.get(name, 0)
            vector_score = vector_matches.get(name, 0)
            # Weighted average
            combined_score = (fuzzy_score * 0.6 + vector_score * 0.4)
            combined_scores[name] = combined_score
        
        # Sort by combined score
        sorted_matches = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Format results
        results = []
        for name, score in sorted_matches:
            results.append({
                "name": name,
                "similarity_score": round(score, 2),
                "fuzzy_score": fuzzy_matches.get(name, 0),
                "vector_score": vector_matches.get(name, 0)
            })
        
        return results

# ============================
# Dataset Preparation (30+ similar names)
# ============================
def prepare_name_dataset() -> List[str]:
    """
    Create a dataset of similar names (groups of similar names)
    """
    name_groups = [
        # English names with variations
        ["Catherine", "Katharine", "Kathryn", "Cathy", "Katherine", "Kate", "Katie", "Cat"],
        ["Elizabeth", "Elisabeth", "Liz", "Lizzie", "Beth", "Eliza", "Liza", "Betty"],
        ["Christopher", "Chris", "Christoph", "Kristoffer", "Kristopher", "Topher"],
        ["Jonathan", "John", "Jon", "Nathan", "Nate", "Johnny"],
        
        # Indian names with variations
        ["Geetha", "Gita", "Geeta", "Geethu", "Gitu", "Geethika"],
        ["Suresh", "Suresh", "Sureshwar", "Sureshbabu", "Suresh Kumar"],
        ["Priyanka", "Priyank", "Priya", "Priyanshi", "Priyanshu"],
        ["Mohammed", "Muhammad", "Mohammad", "Md", "Mohd", "Muhammed"],
        
        # Short forms and nicknames
        ["Alexander", "Alex", "Alec", "Sasha", "Xander"],
        ["William", "Will", "Bill", "Billy", "Liam", "Willy"],
        ["Robert", "Rob", "Bob", "Bobby", "Robbie"],
        ["Richard", "Rick", "Dick", "Rich", "Ricky"],
        
        # Phonetic variations
        ["Sean", "Shawn", "Shaun", "Shon"],
        ["Steven", "Stephen", "Steve", "Stefan"],
        ["Brian", "Bryan", "Bryon"],
        ["Alan", "Allen", "Allan", "Alen"],
        
        # International variations
        ["Sofia", "Sophia", "Sofiya", "Sofya"],
        ["Ana", "Anna", "Hannah", "Hana"],
        ["Michael", "Micheal", "Miguel", "Mikhail"],
        ["David", "Dave", "Davy", "Dafydd"],
        
        # Additional names to reach 30+
        ["Rahul", "Raoul", "Raul"],
        ["Anand", "Ananda", "Anand", "Anand Kumar"],
        ["Deepak", "Dipak", "Deep", "Dipak"],
        ["Vikram", "Vikramaditya", "Vicky"],
        ["Arun", "Arun Kumar", "Aroon"],
        ["Sunita", "Suneeta", "Sunitha"],
        ["Ramesh", "Rameshwar", "Ramesh Babu"],
        ["Anjali", "Anjali", "Anjali Kumari"],
        ["Manish", "Maneesh", "Manish Kumar"]
    ]
    
    # Flatten the list and remove exact duplicates while preserving order
    all_names = []
    seen = set()
    for group in name_groups:
        for name in group:
            if name not in seen:
                seen.add(name)
                all_names.append(name)
    
    return all_names

# ============================
# Main Execution
# ============================
def main():
    # Prepare dataset
    print("Preparing name dataset...")
    names_dataset = prepare_name_dataset()
    print(f"Total names in dataset: {len(names_dataset)}")
    print(f"Sample names: {names_dataset[:10]}...\n")
    
    # Initialize matchers
    print("Initializing name matchers...")
    fuzzy_matcher = FuzzyNameMatcher(names_dataset)
    vector_matcher = VectorNameMatcher(names_dataset)
    hybrid_matcher = HybridNameMatcher(names_dataset)
    
    # Interactive mode
    print("\n" + "="*60)
    print("NAME MATCHING SYSTEM")
    print("="*60)
    print("\nEnter 'quit' to exit the program")
    
    while True:
        print("\n" + "-"*60)
        user_input = input("\nEnter a name to find similar matches: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter a name.")
            continue
        
        print(f"\nSearching for: '{user_input}'")
        
        # Get matches using hybrid approach (recommended)
        print("\n" + "="*60)
        print("HYBRID MATCHING RESULTS (Recommended)")
        print("="*60)
        
        # Best match
        best_result = hybrid_matcher.find_best_match(user_input)
        print(f"\n BEST MATCH:")
        print(f"   Name: {best_result['best_match']}")
        print(f"   Similarity Score: {best_result['similarity_score']}%")
        print(f"   Method Used: {best_result['method_used']}")
        
        # Ranked matches
        print(f"\n TOP 10 SIMILAR NAMES:")
        ranked_matches = hybrid_matcher.find_ranked_matches(user_input, 10)
        
        for i, match in enumerate(ranked_matches, 1):
            print(f"{i:2}. {match['name']:30} Score: {match['similarity_score']:6.2f}% "
                  f"(Fuzzy: {match['fuzzy_score']:.0f}, Vector: {match['vector_score']:.0f})")
        
        # Optional: Show individual method results for comparison
        print("\n" + "-"*60)
        print("COMPARISON OF DIFFERENT METHODS")
        print("-"*60)
        
        # Fuzzy matching results
        fuzzy_best, fuzzy_score = fuzzy_matcher.find_best_match(user_input)
        print(f"\n Fuzzy Matching Best Match: {fuzzy_best} ({fuzzy_score}%)")
        
        # Vector matching results
        vector_best, vector_score = vector_matcher.find_best_match(user_input)
        print(f" Vector Matching Best Match: {vector_best} ({vector_score}%)")
        
        # Show matches above threshold
        print(f"\n High Confidence Matches (Above 80% similarity):")
        high_conf_matches = fuzzy_matcher.find_matches_with_threshold(user_input, 80)
        if high_conf_matches:
            for name, score in high_conf_matches[:5]:  # Show top 5
                print(f"   • {name}: {score}%")
        else:
            print("   No matches above 80% similarity threshold")

def batch_test():
    """
    Test the system with multiple input names
    """
    print("\n" + "="*60)
    print("BATCH TESTING THE SYSTEM")
    print("="*60)
    
    names_dataset = prepare_name_dataset()
    hybrid_matcher = HybridNameMatcher(names_dataset)
    
    test_cases = [
        "Geetha",
        "Katherine",
        "Mohammed",
        "Chris",
        "Sofia",
        "Rahul",
        "Priya",
        "Alex",
        "Steven",
        "Anand"
    ]
    
    for test_name in test_cases:
        print(f"\nTest: '{test_name}'")
        result = hybrid_matcher.find_best_match(test_name)
        print(f"   Best Match: {result['best_match']} ({result['similarity_score']}%)")
        
        # Get top 3 matches
        matches = hybrid_matcher.find_ranked_matches(test_name, 3)
        print(f"   Top 3: {', '.join([m['name'] for m in matches])}")

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.call(['pip', 'install', 'fuzzywuzzy', 'python-Levenshtein', 'scikit-learn', 'pandas'])
        print("Packages installed. Please run the script again.")
        exit()
    
    # Run the system
    main()
    
    # Uncomment to run batch tests
    # batch_test()
