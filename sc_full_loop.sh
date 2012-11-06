LOG_FILE=log/status

for ((i=0;i<5;i++));
do 
if (($i==0))
then cp $LOG_FILE $LOG_FILE.$((i+1));
else
cp $LOG_FILE.$i $LOG_FILE.$((i+1));
fi;
done;

# patch extraction
#python2 get_random_patches_from_non_annotated.py -c inriaPedestrian.config
#echo $(date): patches done > $LOG_FILE

## clustering
#python2 cluster_with_kmeans.py -c inriaPedestrian.config;
#echo $(date): clustering done > $LOG_FILE
 
## feature computation
#python2 compute_features_from_annotations.py -c inriaPedestrian.config;
#echo $(date): positive features >> $LOG_FILE
#python2 compute_features_from_non_annotated.py -c inriaPedestrian.config --data-split 100 ; 
#echo $(date): negative features >> $LOG_FILE

## svm
#python2 svm_training.py -c inriaPedestrian.config --data-split 100
#echo $(date): svm training >> $LOG_FILE

## bootstrap feature harvesting
#python2 compute_features_from_non_annotated.py -b1 -c inriaPedestrian.config --data-split 100 ; 
#echo $(date): bootstrap features >> $LOG_FILE

## svm-2
#python2 svm_training.py -c inriaPedestrian.config --data-split 100 -b1
#echo $(date): bootstrap svm >> $LOG_FILE

# classification
#python2 classify_annotations.py -c inriaPedestrian.config -s validation -b1
#echo $(date): classify annotations >> $LOG_FILE
python2 classify_non_annotated_images.py -c inriaPedestrian.config -s validation -b1 --data-split 50
echo $(date): classify non annotated >> $LOG_FILE
