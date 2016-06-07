# Generate input.txt file from corpus
python Reader.py -gen_bfile --outdir output

# Clusters input.txt into 50 clusters:
./brown-cluster/wcluster --text output/plain.txt --c 50

# Move output files from default directory to ouput
# Format [Code, word, cluster_num]
mv plain*/* output/
rmdir plain*

# Generate feature file
python create_pos_word_feats.py --brown_filename output/paths --output_filename output/features.txt
