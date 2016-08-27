## Ricardo A. Calix, PNW, 2016
## Python code to perform vector arithmetic and simple clustering tasks

import numpy as np
import math
import operator
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity


##########################################################

#list_meds = ["prozac","zoloft","ativan","morphine","valium","adderall","vicodin","ambien","percocet","prednisone","xanax","tramadol"]
list_meds = ["echinacea","valerian","melatonin"]
list_effects = ["high","feel", "good","anxiety","pain","sleep","headache","depression","sick"]


###########################################################

dictionaryWordsToVectors = {}
##dictionaryWordsToVectors["age"]= "[2,3,4]"

###########################################################
words_to_plot = []
def func_words_to_plot():
    f_plot = open("words_to_plot.txt","r")
    for word in f_plot.readlines():
        words_to_plot.append(word)

    f_plot.close()

###########################################################

f = open("data/yong_bing_word2vec_vector_space.txt","r")
#f = open("data/mars_word2vec_vector_space.txt","r")
#f = open("word2vec_vector_space_new100_2021_arithmetic_2d.txt","r")
#f = open("word2vec_vector_space_100words.txt","r")
#f = open("2021_yes_tweets_vector_space_4090words.txt","r")
#f = open("toy_test_data.txt","r")
for line in f.readlines():
    #print line
    features = line.split(",")
    #print features
    n = len(features)
    vector = features[1:n-1]
    #print "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"
    #print vector
    vector_float = [float(x) for x in vector]
    #print "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"
    #print vector_float
    #x = raw_input()
    key_word = features[0]
    dictionaryWordsToVectors[key_word] = vector_float





#print dictionaryWordsToVectors["xanax"]
    
###########################################################

def dot_product2(v1,v2):
    return sum(map(operator.mul, v1,v2))

###########################################################
def vector_cos5(v1,v2):
    prod = dot_product2(v1,v2)
    len1 = math.sqrt(dot_product2(v1,v1))
    len2 = math.sqrt(dot_product2(v2,v2))
    return prod / (len1 * len2)


############################################################

def getVectorFromWord(word):
    vector = dictionaryWordsToVectors[word]
    #print vector
    return vector


###########################################################

def findClosestWordToThisVector_cosine_list(vector):
    top_5_closest = []
    temp_dict = {}
    for key, value in dictionaryWordsToVectors.iteritems():
        #distance_result = distance.euclidean(vector,value)
        distance_result = vector_cos5(vector, value)
        temp_dict[key]=distance_result
    sorted_x = sorted(temp_dict.items(),key=operator.itemgetter(1))
    #print sorted_x
    
    for item in sorted_x:
        if item[1] < 1 and item[1] > 0.8:
            #print item[1]
            #x = raw_input()
            top_5_closest.append(item[0])
    return top_5_closest




###########################################################

def findClosestWordToThisVector_list(vector):
    top_5_closest = []
    temp_dict = {}
    for key, value in dictionaryWordsToVectors.iteritems():       
        #distance_result = vector_cos5(vector, value)
        distance_result = distance.euclidean(vector,value)
        temp_dict[key]=distance_result
    sorted_x = sorted(temp_dict.items(),key=operator.itemgetter(1))
    #print sorted_x
    for i in range(20):
        top_5_closest.append(sorted_x[i][0])
    return top_5_closest


############################################################

def findClosestWordToThisVector(vector):
    ##use cosine distance function? do this in matrix format for efficiency
    shortest_distance = 1000000.0
    shortest_word = "null_word"
    for key, value in dictionaryWordsToVectors.iteritems():
        #distance_result = distance.euclidean(vector,value)
        #distance_result = cosine_similarity(vector,value)
        distance_result = vector_cos5(vector, value)

        #print distance_result
        #print key
        #print shortest_distance
        #print shortest_word
        #x = raw_input()
        if distance_result < shortest_distance:
            shortest_distance = distance_result
            shortest_word = key
    print shortest_distance
    return shortest_word

###########################################################

#a1 = np.array([0,1,2])
#b1 = np.array([3,4,5])
#result = a1 + b1
#print result

###########################################################

a = np.array(getVectorFromWord("prozac"))
b = np.array(getVectorFromWord("prozac"))
c = np.array(getVectorFromWord("morphine"))

offset = np.array([-150, 0])

#print a
#print "WWWWWWWWWWWWWWWWWWWWWWW"
#print b
#print "WWWWWWWWWWWWWWWWWWWWWWW"
#print c
#print "WWWWWWWWWWWWWWWWWWWWWWW"


## king + queen - man

#result_vector = a + offset 
result_vector = a
#result_vector = np.add(a,b) #np.subtract(np.add(a,b),c)

print result_vector



result_word = findClosestWordToThisVector_list(result_vector)
print result_word

##########################################################

##zoloft, ativan

def print_clusters(list_meds):
    for med in list_meds:
        a = np.array(getVectorFromWord(med))
        result_vector = a

        result_word = findClosestWordToThisVector_list(result_vector)
        #result_word = findClosestWordToThisVector_cosine_list(result_vector)

        print med
        print result_word

############################################################

print_clusters(list_meds)

##########################################################

def semantic_relationships():
    #anxiety
    for med in list_meds:
        for effect in list_effects:
            med_vector = np.array(getVectorFromWord(med))
            effect_vector = np.array(getVectorFromWord(effect))
            result_vector = med_vector + effect_vector
            result_words = findClosestWordToThisVector_list(result_vector)
            print "med:" + med + " + " + "effects:" + effect + " -->> " + " , ".join(result_words)
            ##############
            result_vector = med_vector - effect_vector
            result_words = findClosestWordToThisVector_list(result_vector)
            print "med:" + med + " - " + "effects:" + effect + " -->> " + " , ".join(result_words)


##########################################################

semantic_relationships()

##########################################################

print "<<<<DONE>>>>"
