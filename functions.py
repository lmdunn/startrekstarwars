from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

global gs

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



def lemmatize_text(text):
    split_text = text.split()
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in split_text])

def stem_text(text):
    split_text = text.split()
    p_stemmer = PorterStemmer()
    return ' '.join([p_stemmer.stem(word) for word in split_text])
