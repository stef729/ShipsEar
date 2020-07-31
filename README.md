# ShipsEar

Process the ShipsEar dataset, the dataset includes 90 audio files,  includes 5 classes and 12 categories.
We preprocessed the data by removing the blank signal and framed the original signal by 5s. 
Finally, 1956 labeled sound samples with a total of 12 categories of targets can be used. 

Category	Targets	Frames
Class A	fishing boats, trawlers, mussel boats, tugboats, dredgers	98/28/ /95/23/52
Class B	motorboats, pilot boats, sailboats	195/26/76
Class C	passenger ferries	703
Class D	ocean liners, ro-ro vessels	174/261
Class E	background noise	225

The 'Preprocessing' achieve the function of framing, visiualization, add noise, feature selection
The 'ShipsEar_Classification_CNN,LSTM,CRNN' verify the classification performance of different network models
The 'ShipsEar_Classification_ResNet' is the transfer learning
The 'ShipsEar_Classification_VGGish' Verify Google's VGGish network classification performance
The 'ShipsEar_data_aug 5s/1s' and 'specaugment' verify the data augmentation with methods of google's specAugment and network structure with VGGish

The classification performance can reach 97%, but the division of training set and test set is not independent, it does not mean that the method is effective.
In the future, will continue to verify.
