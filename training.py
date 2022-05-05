from pykeen.triples import TriplesFactory

from pykeen.pipeline import pipeline

training_set = "datasets/FB15K-237.2/Release/train.txt"
testing_set = "datasets/FB15K-237.2/Release/test.txt"
val_set = "datasets/FB15K-237.2/Release/valid.txt"

result = pipeline(
    training=training_set,
    testing=testing_set,
    validation=val_set,
    model='TransE',
    epochs=400,  # short epochs for testing
)
print(result)
result.save_to_directory("results_400")
