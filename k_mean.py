"""
To-Do:

1. Use Word2Vec to get average word embeddings for each ERROR_INFO
2. Try K-means on the Word2Vec and see if it yields better results
3. Present the results to Robert to examine performance
"""

# import the necessary libraries 
import pandas as pd 
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer 
from helper import all_clean_error_info, get_embeddings_from_row_non_zero_avg, produce_wcss_diagram_in_range

def main(): 
    df=pd.read_json(r'Data/error_logs_user2.json') 
    
    # Extract the sentence only 
    sentence = all_clean_error_info(df["ERROR_INFO"])

    # # create vectorizer 
    vectorizer = TfidfVectorizer(stop_words=["at"])
  
    # vectorizer the text documents 
    vectorized_documents = vectorizer.fit_transform(sentence)

    # embeddings = vectorized_documents  # Word by word focus
    embeddings = np.asarray(vectorized_documents.mean(axis=1))  # Average the word embeddings for each document diving by total length including zero values
    # embeddings = get_embeddings_from_row_non_zero_avg(vectorized_documents)

    # produce_wcss_diagram_in_range(range(1,10), embeddings, 'wcss_elbow_plot.png')  # Produce the elbow plot

    # After running the above code, the optimal number of clusters is 3
    # Because the variance is significantly lower than other method, we will use average of word embeddings including zeros when diving.
    # Cluster size = 3
    kmeans = KMeans(n_clusters=3, random_state=2002)

    kmeans.fit(embeddings)
    labels = kmeans.labels_

    # Add cluster labels to the original dataframe
    df['cluster'] = labels

    # Compute Within-Cluster Sum of Squares (WCSS)
    wcss = kmeans.inertia_
    
    # Print the evaluation metrics
    print(f"WCSS (Within-Cluster Sum of Squares): {wcss:.10f}")

    # Prepare data for JSON
    clustered_data = {}
    for i, cluster in enumerate(df['cluster'].unique()):
        clustered_data[i] = df[df['cluster'] == cluster].drop(columns=['cluster'])["ERROR_INFO"].tolist()

    # Save to JSON file
    with open('clustered_data2.json', 'w') as f:
        json.dump(clustered_data, f, indent=4)


if __name__ == "__main__":
    main()