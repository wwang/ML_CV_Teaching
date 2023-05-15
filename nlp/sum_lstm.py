# supporting functions for teaching an LSTM for summing two integers

import numpy as np


# append spaces so that every entry in data has "leng" characters
def append_spaces(data, leng):
  out = []
  for t in data:
    t2 = str(t)
    t2 += ''.join([' ' for padding in range(leng - len(t))])
    out.append(t2)

  return out

# Converts x or y arrays of strings into array of char indices.
def dataset_to_indices(data, vocabulary):
    out = []
    
    char_to_index = {char: index for index, char in enumerate(vocabulary)}
    
    for example in data:
        example_encoded = [char_to_index[char] for char in example]
        out.append(example_encoded)
        
    return out

# Convert x or y sets of char indices into one-hot vectors.
def dataset_to_one_hot(data, vocabulary):
    out = []
    
    for example in data:
        pattern = []
        for index in example:
            vector = [0 for _ in range(len(vocabulary))]
            vector[index] = 1
            pattern.append(vector)
        out.append(pattern)            
        
    return out

# One function to convert x or y arrays of string into one-hot vector
def string_to_one_hot(data, vocabulary):
  out = dataset_to_indices(data, vocabulary)
  out = dataset_to_one_hot(out, vocabulary)

  return np.array(out)

# Converts a sequence (array) of one-hot encoded vectors back into the string based on the provided vocabulary.
def devectorization(sequence, vocabulary):
    index_to_char = {index: char for index, char in enumerate(vocabulary)}
    strings = []
    for char_vector in sequence:
        char = index_to_char[np.argmax(char_vector)]
        strings.append(char)
    return ''.join(strings)


# Train data set verification
def dataset_verification(x, y):
  print('x.shape: ', x.shape) # (input_sequences_num, input_sequence_length, supported_symbols_num)
  print('y.shape: ', y.shape) # (output_sequences_num, output_sequence_length, supported_symbols_num)

  # How many characters each summation expression has.
  input_sequence_length = x.shape[1]

  # How many characters the output sequence of the RNN has.
  output_sequence_length = y.shape[1]

  # The length of one-hot vector for each character in the input (should be the same as vocabulary_size).
  supported_symbols_num = x.shape[2]

  # The number of different characters our RNN network could work with (i.e. it understands only digits, "+" and " ").
  vocabulary_size = len(vocabulary)

  print('input_sequence_length: ', input_sequence_length)
  print('output_sequence_length: ', output_sequence_length)
  print('supported_symbols_num: ', supported_symbols_num)
  print('vocabulary_size: ', vocabulary_size)
