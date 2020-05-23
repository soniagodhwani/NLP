import sys
import numpy
import torch
import torch.nn as nn 
import argparse
import time
import pickle

T = 0
USC_EMAIL = 'youremail@usc.edu'  # TODO(student)

class DatasetReader(object):
  # TODO(student): You must implement this.
  @staticmethod
  def ReadFile(filename, term_index, tag_index):
    """Reads file into dataset, while populating term_index and tag_index.
   
    Args:
      filename: Path of text file containing sentences and tags. Each line is a
        sentence and each term is followed by "/tag". Note: some terms might
        have a "/" e.g. my/word/tag -- the term is "my/word" and the last "/"
        separates the tag.
      term_index: dictionary to be populated with every unique term (i.e. before
        the last "/") to point to an integer. All integers must be utilized from
        2 to number of unique terms+1, without any gaps nor repetitions.
        1 is reserved for unknown words.
      tag_index: same idea as term_index, but for tags. Start at index 1. Assume
        no unknown tags in testing.

    Return:
      The parsed file as a list of lists: [parsedLine1, parsedLine2, ...]
      each parsedLine is a list: [(term1, tag1), (term2, tag2), ...] 
    """
    file_obj = open(filename,"r")
    lines = file_obj.readlines()
    parsedlines = []

    term_id = len(term_index) + 1
    tag_id = len(tag_index) + 1
    for line in lines:
      line = line[:-1]
      term_tag_list = line.split(' ')
      parsedline = []
      for term_tag in term_tag_list:
        tag_sindex = term_tag.rindex("/")
        term = term_tag[:tag_sindex]
        tag = term_tag[tag_sindex+1 :]
        if term not in term_index:
          term_index[term] = term_id
          term_id += 1
        if tag not in tag_index:
          tag_index[tag] = tag_id
          tag_id += 1
        parsedline.append((term_index[term],tag_index[tag]))
      parsedlines.append(parsedline)
    return parsedlines

  # TODO(student): You must implement this.
  @staticmethod
  def BuildMatrices(dataset):
    """Converts dataset [returned by ReadFile] into numpy arrays for tags, terms, and lengths.

    Args:
      dataset: Returned by method ReadFile. It is a list (length N) of lists:
        [sentence1, sentence2, ...], where every sentence is a list:
        [(word1, tag1), (word2, tag2), ...], where every word and tag are integers.

    T is the maximum length. You can define this as a global variable. You will use it when you create 
    a SequenceModel class later.

    Returns:
      Tuple of 3 numpy arrays: (terms_matrix, tags_matrix, lengths_arr)
        terms_matrix: shape (N, T) int64 numpy array. Row i contains the word
          indices in dataset[i].
        tags_matrix: shape (N, T) int64 numpy array. Row i contains the tag
          indices in dataset[i].
        lengths: shape (N) int64 numpy array. Entry i contains the length of
          sentence in dataset[i].

      For example, calling as:
        BuildMatrices([[(1,2), (4,10)], [(13, 20), (3, 6), (7, 8), (3, 20)]])
      i.e. with two sentences, first with length 2 and second with length 4,
      should return the tuple:
      (
        [[1, 4, 0, 0],    # Note: 0 padding.
         [13, 3, 7, 3]],

        [[2, 10, 0, 0],   # Note: 0 padding.
         [20, 6, 8, 20]], 

        [2, 4]
      )
    """
    global T
    T = len(max(dataset, key=len))
    N = len(dataset)
    terms_matrix = numpy.empty([N, T], dtype=int)
    tags_matrix = numpy.empty([N,T], dtype = int)
    lengths = numpy.empty([N],dtype = int)

    for i in range(len(dataset)):
      line = dataset[i]
      terms = []
      tags = []
      for word_tag in line:
        terms.append(word_tag[0])
        tags.append(word_tag[1])
      terms.extend([0] * (T - len(terms)))
      terms_matrix[i] = terms
      tags.extend([0] * (T - len(tags)))
      tags_matrix[i] = tags
      lengths[i] = len(line)

    return (terms_matrix,tags_matrix,lengths)

  @staticmethod
  def ReadData(train_filename, test_filename=None):
    """Returns numpy arrays and indices for train (and optionally test) data.

    NOTE: Please do not change this method. The grader will use an identitical
    copy of this method (if you change this, your offline testing will no longer
    match the grader).

    Args:
      train_filename: .txt path containing training data, one line per sentence.
        The data must be tagged (i.e. "word1/tag1 word2/tag2 ...").
      test_filename: Optional .txt path containing test data.

    Returns:
      A tuple of 3-elements or 4-elements, the later iff test_filename is given.
      The first 2 elements are term_index and tag_index, which are dictionaries,
      respectively, from term to integer ID and from tag to integer ID. The int
      IDs are used in the numpy matrices.
      The 3rd element is a tuple itself, consisting of 3 numpy arrsys:
        - train_terms: numpy int matrix.
        - train_tags: numpy int matrix.
        - train_lengths: numpy int vector.
        These 3 are identical to what is returned by BuildMatrices().
      The 4th element is a tuple of 3 elements as above, but the data is
      extracted from test_filename.
    """
    term_index = {'__oov__': 1}  # Out-of-vocab is term 1.
    tag_index = {}
    
    train_data = DatasetReader.ReadFile(train_filename, term_index, tag_index)
    train_terms, train_tags, train_lengths = DatasetReader.BuildMatrices(train_data)
    
    if test_filename:
      test_data = DatasetReader.ReadFile(test_filename, term_index, tag_index)
      test_terms, test_tags, test_lengths = DatasetReader.BuildMatrices(test_data)

      if test_tags.shape[1] < train_tags.shape[1]:
        diff = train_tags.shape[1] - test_tags.shape[1]
        zero_pad = numpy.zeros(shape=(test_tags.shape[0], diff), dtype='int64')
        test_terms = numpy.concatenate([test_terms, zero_pad], axis=1)
        test_tags = numpy.concatenate([test_tags, zero_pad], axis=1)
      elif test_tags.shape[1] > train_tags.shape[1]:
        diff = test_tags.shape[1] - train_tags.shape[1]
        zero_pad = numpy.zeros(shape=(train_tags.shape[0], diff), dtype='int64')
        train_terms = numpy.concatenate([train_terms, zero_pad], axis=1)
        train_tags = numpy.concatenate([train_tags, zero_pad], axis=1)

      return (term_index, tag_index,
              (train_terms, train_tags, train_lengths),
              (test_terms, test_tags, test_lengths))
    else:
      return term_index, tag_index, (train_terms, train_tags, train_lengths)


class SequenceModel(nn.Module):
  def __init__(self,
  	num_terms=1000, num_tags=40, max_length=310, hidden_size=300, 
  ):
    """Constructor. You can add code but do not remove any code.

    The arguments are arbitrary: when you are training on your own, PLEASE set
    them to the correct values (e.g. from main()).

    Args:
      max_lengths: maximum possible sentence length.
      num_terms: the vocabulary size (number of terms).
      num_tags: the size of the output space (number of tags).
      hidden_size: the size of the hidden vectors.

    You will be passed these arguments by the grader script.
    """

    super(SequenceModel, self).__init__()

    self.hidden_size = hidden_size

    self.input2hidden = nn.Linear(num_terms + hidden_size, hidden_size)
    self.input2output = nn.Linear(num_terms + hidden_size, num_tags)
    self.softmax = nn.LogSoftmax(dim=1)

  # TODO(student): You must implement this.
  def save_model(self, filename):
    """Saves model to a file."""
    pass

  # TODO(student): You must implement this.
  def load_model(self, filename):
    """Loads model from a file."""
    pass

  # TODO(student): You may implement this if you choose.
  def build(self):
    """Prepares the class for training.
    
    It is up to you how you implement this function, as long as forward
    works.
    """
    pass

  ## TODO(student): You must implement this.
  def forward(self, input_vector, hidden_vector):
    """This is a necessary component of PyTorch's nn.Module. It defines the forward pass of
    the neural network, and is used by the library to auto-differentiate and derive the 
    backward pass (also known as back-propagation).
    
    It is up to you how you implement this function.
    """
    pass 

# TODO(student): You must implement this.
def train(model, terms, tags, lengths, batch_size=32, learn_rate=1e-7):
  """Performs updates on the model given training data.

  This will be called with numpy arrays similar to the ones created in ReadData
  Args:
    terms: int64 numpy array of size (# sentences, max sentence length)
    tags: int64 numpy array of size (# sentences, max sentence length)
    lengths: lengths of each term sequence.
    batch_size: int indicating batch size. Grader script will not pass this,
      but it is only here so that you can experiment with a "good batch size"
      from your main block.
    learn_rate: float for learning rate. Grader script will not pass this,
      but it is only here so that you can experiment with a "good learn rate"
      from your main block.
  """
  pass

# TODO(student): You must implement this.
def evaluate(model, terms, lengths):
  """Performs updates on the model given training training data.

  This will be called with numpy arrays similar to the ones created in ReadData
  Args:
    terms: int64 numpy array of size (# sentences, max sentence length)
    lengths: lengths of each term sequence.

  Returns:
    predicted_tags: int64 numpy array of size (# sentences, max sentence length)
  """
  pass


def main(args):
  # Read dataset.
  train_filename = args.i
  test_filename = train_filename.replace('_train_', '_dev_')
  term_index, tag_index, train_data, test_data = DatasetReader.ReadData(train_filename, test_filename)
  (train_terms, train_tags, train_lengths) = train_data
  (test_terms, test_tags, test_lengths) = test_data

  model = SequenceModel() # <student fills in>
  model.build()
  start_time = time.time()
  for j in xrange(args.e):
    train(model, train_terms, train_tags, train_lengths)
    print('Finished epoch %i. Evaluating ...' % (j+1))
    predicted_tags = evaluate(model, test_terms, test_lengths)
    if (time.time() - start_time) > args.t:
      break
  model.save_model(args.m)
  pickle.dump(predicted_tags, open(args.o, 'wb'))

# if __name__ == '__main__':
#   parser = argparse.ArgumentParser(description='Train RNN.')
#   parser.add_argument('-i', type=str, help='training file path.')
#   parser.add_argument('-m', type=str, help='Model output path.')
#   parser.add_argument('-o', type=str, help='Predictions output path.')
#   parser.add_argument('-t', type=float, help='Length of time training (seconds).')
#   parser.add_argument('-e', type=int, help='Num epochs.')
#   args = parser.parse_args()
#
#   main(args)

a = {}
b = {}
x=  DatasetReader.ReadFile("test.txt",a,b)
print a
print b

print DatasetReader.BuildMatrices(x)