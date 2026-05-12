import sys
sys.path.append('src')
from evaluator.prompts import clean_concepts, extract_keywords_from_text, generate_template_question

# Test concept cleaning
test_concepts = ['run', 'jump', 'machine learning', 'neural network', 'algorithm', 'data', 'process', 'system']
cleaned = clean_concepts(test_concepts)
print('Cleaned concepts:', cleaned)

# Test keyword extraction
test_text = 'Machine learning algorithms process data using neural networks to make predictions.'
keywords = extract_keywords_from_text(test_text)
print('Extracted keywords:', keywords)

# Test question generation with valid concepts
question1 = generate_template_question('what', ['machine learning', 'neural network'], 'Data Processing', 'Process data using algorithms')
print('Question with valid concepts:', question1)

# Test question generation with invalid concepts (should fallback)
question2 = generate_template_question('what', ['run', 'jump', 'data', 'process'], 'Data Processing', 'Process data using algorithms')
print('Question with invalid concepts (fallback):', question2)