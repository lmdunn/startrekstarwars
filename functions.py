import pandas as pd
import requests
import time
from statistics import mean
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

##
def build_reddit_df(subreddit, desired_size, endpoint = '/reddit/search/submission', limit = 100):
    
    '''
    NOTE: you must set your desired dataframe equal to this function to save the dataframe outside the function.
    
    This function is designed to use pushshift API to build a dataframe of specified size 
    filled with data from the specified subreddit. It starts with the most recent post and works backwards.

    subreddit = the subreddit you'd like to scrape
    
    desired_size = the total number of posts you'd like to have in the dataframe. The function will hit the minimum size 
        above that value that the 'limit' value allows. In other words, it may go over this value up to the amount of the limit.
    
    endpoint = your desired endpoint. Defaults to '/reddit/search/submission' for submissions (main post) and
       '/reddit/search/comment' for comments at the time of the writing of this function (6/22/2022)     

    limit = the limit for number of posts that can be pulled at once. The default is 100, the maximum
        allowed at the time of the writing of this function (6/22/2022)
    '''
    
    url = 'https://api.pushshift.io'+endpoint
    
    counter = 0
    
    fncdf = pd.DataFrame() #establish with certainty that the new dataframe name is empty.
    
    for i in range(2):
   
        if len(fncdf) == 0:
            params = {
                'subreddit': subreddit,
                'size': limit,
                'filter': ['title', 'selftext', 'subreddit', 'created_utc'] #katie pointed out this parameter to me to save cleaning later.
            }
            res = requests.get(url, params)
            if res.status_code == 200:
                data = res.json()
                posts = data['data']
                fncdf = pd.DataFrame(posts)
                counter += 1
            else:
                print(f'ERROR: status code not 200. Failure occured on loop number {counter+1}')

        else: # after the df has been established.
            while len(fncdf) < desired_size:
                params = {
                    'subreddit': subreddit,
                    'size': limit,
                    'before': fncdf.iloc[-1]['created_utc'],
                    'filter': ['title', 'selftext', 'subreddit', 'created_utc']
                }
                res = requests.get(url, params)

                if res.status_code == 200:
                    data = res.json()
                    posts = data['data']
                    newdf = pd.DataFrame(posts)
                    fncdf = pd.concat([fncdf, newdf], ignore_index = True)
                    counter +=1
                    time.sleep(3) #alanna suggested adding this

                else:
                    print('ERROR: status code not 200. Failure occured on loop number {counter+1}')
    
    return fncdf

##
##

def clean_subreddit_df(dataframe):
    print('Initial number of duplicate titles:', len(dataframe[dataframe.duplicated(['title'])]))
    print('*'*25)
    print(f'Initial Shape: {dataframe.shape}')
    print('='*20)
    print(f'Initial Top 5 Value Counts: {dataframe["selftext"].value_counts().head()}')
    dataframe = pd.concat([dataframe[dataframe['selftext']=='[removed]'],
                         dataframe[dataframe['selftext']==''],
                         dataframe[dataframe['selftext']=='[deleted]'],
                        dataframe[(dataframe['selftext'] != '[removed]') & (dataframe['selftext'] != '') & (dataframe['selftext'] != '[deleted]')]\
                          .drop_duplicates(["selftext"], keep = 'first')])
    dataframe = dataframe.drop_duplicates(['title'], keep = 'first')
    print('')
    print('='*20)
    print('')
    print(f'Final Shape: {dataframe.shape}')
    print('='*20)
    print(f'Final Top 5 Value Counts: {dataframe["selftext"].value_counts().head()}')
    print('*'*25)
    print('Final number of duplicate titles:', len(dataframe[dataframe.duplicated(['title'])]))
    return dataframe

##
##

# find the mean word length
def mean_word_length (string):
    
#split the string
    word_list = [i for i in string.split()]

#remove the characters from the string
    cleaned_words = []
    for word in word_list:
        cleaned_word = ''.join([i for i in word if i.isalpha()])
        cleaned_words.append(cleaned_word)

    return mean([len(i) for i in cleaned_words])

##
##

def pipe_grid(pipe_params, grid_params):
    '''
    This function is designed to streamline gridsearching.
    It returns a gridsearch named 'gs'
    'pipe_params' should be a list of tuples consisting of a series of name/transform pairs followed by a name/model, 
            e.g. [('cvec', CountVectorizer()), ('log', LogisticRegression())]
    'grid_params' should be a series of parameters for those transforms and the model in the form of a dictionary,
            e.g. {'cvec__ngram_range': [(1,1), (1,2)]}
    Be sure the names for the 'pipe_params' and in the 'grid_params match'
    '''
    
    global gs
    
    pipe = Pipeline(pipe_params)
    
    gs = GridSearchCV(pipe, grid_params)
   
    return gs

def pipe_grid_njobs(pipe_params, grid_params):
    '''
    This function is designed to streamline gridsearching. This version sets n_jobs = -1
    It returns a gridsearch named 'gs'
    'pipe_params' should be a list of tuples consisting of a series of name/transform pairs followed by a name/model, 
            e.g. [('cvec', CountVectorizer()), ('log', LogisticRegression())]
    'grid_params' should be a series of parameters for those transforms and the model in the form of a dictionary,
            e.g. {'cvec__ngram_range': [(1,1), (1,2)]}
    Be sure the names for the 'pipe_params' and in the 'grid_params match'
    '''

    global gs
    
    pipe = Pipeline(pipe_params)
    
    gs = GridSearchCV(pipe, grid_params, n_jobs = -1)
    
    
    return gs

##
##

# I used the NLP Practice breakfast hour as a model for these functions.

def lemmatize_text(text):
    split_text = text.split()
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in split_text])

def stem_text(text):
    split_text = text.split()
    p_stemmer = PorterStemmer()
    return ' '.join([p_stemmer.stem(word) for word in split_text])
