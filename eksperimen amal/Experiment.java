package tubes2ai;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Experiment {
    static int countExp = 0;
    static int maxTrial = 20;
    int idx;
    Instances data;
    Trial[] trial = new Trial[maxTrial];
    int bestTrialIdx;
    int numTrial;
    
    public Experiment(String filename) throws Exception{
        DataSource source = new DataSource(filename);
        data = source.getDataSet();
        idx = countExp++;
        numTrial = 0;
        if (idx==2) {
            data.setClassIndex(0);
        } else {
            data.setClassIndex(data.numAttributes()-1);
        }
    }
    
    public void newTrial(Classifier cls, String name) {
        trial[numTrial] = new Trial(data, cls, name);
        numTrial++;
    }
    
    public int getBestTrialIdx() {
        int idx=0;
        for (int i=0; i<numTrial; i++) {
            if (trial[i].pctCorrect > trial[idx].pctCorrect) {
                idx = i;
            }
        }
        return idx;
    }
    
    public void printBestTrail() {
        System.out.println("Best Trial: " + trial[getBestTrialIdx()].clsName);
    }
    
    public void saveModel(Classifier cls, String filename) throws IOException {
        ObjectOutputStream writer = new ObjectOutputStream(new FileOutputStream(filename));
        writer.writeObject(cls);
    }
    
}
