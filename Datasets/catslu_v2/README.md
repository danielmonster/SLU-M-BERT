All preprocessed data is stored in ~/preprocessed directory.

The dataset restructured the original CATSLU dataset by moving part 
(first 2000 entries of weather and first 1000 of video) of the test data 
into training data.
The restructuring is done in hope to balance data of different intents, while
preserving sufficient data for testing.

Current distribution of the four intents in training data is: 
Navigation 2358
Music 1524
Video 1004
Weather 1463

Steps to rerun the preprocessing are listed below:
1. Create empty /processed directory under ~/catslu_test/data & ~/catslu_traindev/data
2. Navigate to ~/catslu_test/src, run convert.py, gen_labels.py, agg_testset.py in order without arguments
3. Repeat step 2 in ~/catslu_traindev/src
4. Navigate to the ~/src, run agg_all.py