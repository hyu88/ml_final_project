'''
Generate processed_full_training_dataset
'''
import py_featurelist_gen
import os

def preprocess_data(filename,num,is_fea=False):

    file1='../data/'+filename

    fp = open(file1, 'r')
    line = fp.readline()

    datas=[]
    while line:
        processedTweet = py_featurelist_gen.process_tweet(line)
        line = fp.readline()
        datas.append(processedTweet)
    #end loop
    fp.close()

    file2=os.path.join('../data','processed_'+filename)
    with open(file2,'w') as fout:
        for line in datas:
            fout.write(line+'\n')

    if is_fea:
        print 'processed training file generated!'
        py_featurelist_gen.gen_feature_list(num)


if __name__=='__main__':
    # filename = 'full_training_dataset.csv'
    filename = 'pre_iphone.csv'
    preprocess_data(filename,num=500,is_fea=False)

