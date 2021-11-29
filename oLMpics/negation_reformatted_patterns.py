# Four patterns
def reformat_stem_pattern1(question_text):
  sent1, sent2 = question_text.split(',')
  sent11, sent12 = sent1.split('[MASK]')
  sent11= ' '.join(reversed(sent11.split(' '))).lower()
  reformatted_text = sent2.lstrip().capitalize() + " " + sent11.lstrip().capitalize() + sent12 +' ?'
  return reformatted_text

def reformat_stem_pattern2(question_text):
  sent1, sent2 = question_text.split(',')
  sent11, sent12 = sent1.split('[MASK]')
  sent11= ' '.join(reversed(sent11.split(' '))).lower()
  reformatted_text = sent2.lstrip().capitalize() + " " + sent11.lstrip().capitalize()  +" really" + sent12 + ' ?'
  return reformatted_text

def reformat_stem_pattern3(question_text):
  sent1, sent2 = question_text.split(',')
  sent1 = sent1.replace('[MASK]','').lstrip()
  reformatted_text = sent1 + " entails" + sent2.lower().replace('.','?')
  return reformatted_text

def reformat_stem_pattern4(question_text):
  sent1, sent2 = question_text.split(',')
  sent1 = sent1.replace('[MASK]','').lstrip()
  reformatted_text = "Sentence 1: " + '"'+ sent1 + '.' +" Sentence 2: " + sent2 + " Is Sentence 1 synonym of Sentence 2 ?"
  return reformatted_text