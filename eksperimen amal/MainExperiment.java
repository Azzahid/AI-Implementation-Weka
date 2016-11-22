package tubes2ai;

import java.io.IOException;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;

public class MainExperiment {
    public static void main(String[] args) throws IOException {
        String[] filenames = new String[] {"Team.arff", "iris.arff", "mush.arff"};
        Classifier[] arrCls = new Classifier[] {new NaiveBayes(), new J48(), new SMO()};
        String[] clsNames = new String[] {"Naive Bayes", "J48         ", "SMO        "};
        int numCls = arrCls.length;
        int numFiles = filenames.length;
        Scanner sc = new Scanner(System.in);
        
        Experiment[] exps = new Experiment[numFiles];
        for (int i=0; i<numFiles; i++) {
            try {
                exps[i] = new Experiment(filenames[i]);
                System.out.println("Experiment on " + filenames[i]);
            } catch (Exception e) {
                System.out.println("eror: " + e);
            }
            for (int j=0; j<numCls; j++) {
                Classifier cls = arrCls[j];      
                exps[i].newTrial(cls, clsNames[j]);
                try {
                    exps[i].trial[j].evaluate(exps[i].data);
                } catch (Exception ex) {
                    Logger.getLogger(MainExperiment.class.getName()).log(Level.SEVERE, null, ex);
                }
                exps[i].trial[j].printStatus();
            }
            exps[i].printBestTrail();
            System.out.print("Enter model filename to save: ");
            exps[i].saveModel(exps[i].trial[exps[i].bestTrialIdx].cls, sc.next());
            System.out.println("Model saved");
            System.out.println();
        }
    }
}
