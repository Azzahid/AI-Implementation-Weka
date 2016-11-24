package NaiveBayes;

//import .*;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.Date;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class TesterBayes {
    private static Instances getInstances(Scanner s,String name) throws Exception {
        
        Instances i = new ConverterUtils.DataSource(name).getDataSet();
        System.out.println();
        for (int j = 0; j < i.numAttributes(); j++ ) {
            System.out.println((j + 1) + ". " + i.attribute(j).name());
        }
        System.out.print("Select class index : ");
        i.setClassIndex(s.nextInt() - 1);
        return i;
    }

    private static Classifier getClassifier(Scanner s, Instances i) throws Exception {
        @SuppressWarnings("UnusedAssignment")
        Classifier c = null;
        System.out.println("\n1. Build model");
        System.out.println("2. Load model");
        System.out.print("Choose : ");
        
        //for(int k=0; k<i.numAttributes();k++){
            //System.out.println(i.numDistinctValues(k));
        //}
        switch(s.nextInt()) {
            case 1 :
                c = new NaiveBayes();
                c.buildClassifier(i);
                break;
            case 2 :
                System.out.printf("\nFile MODEL name : ");
                c = (Classifier) weka.core.SerializationHelper.read(s.next());
                break;
            default :
                System.exit(0);
        }
        return c;
    }

    private static Evaluation getEvaluation(Scanner s, Instances i, Classifier c, String name) throws Exception {
        System.out.println("\n1. Full training");
        System.out.println("2. 10-cross-fold validation");
        System.out.print("Choose : ");
        Evaluation e = new Evaluation(i);
        
        switch(s.nextInt()) {
            case 1 :
                fullTraining(name, c,i);
                break;
            case 2 :
                crossValidation(name, c,i);
                break;
            case 3 :
                splitTest(name, i);
            default :
                System.exit(0);
        }
        return e;
    }

    private static void saveModel(Classifier c, String namafile) throws Exception {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(namafile))) {
            oos.writeObject(c);
            oos.flush();
        }
    }

        private static void fullTraining(String name, Classifier c, Instances i) throws Exception {
        //c.buildClassifier(i);
        Evaluation e = new Evaluation(i);
        e.evaluateModel(c, i);
        String namafile = String.format("%s_%s_FULL-TRAINING_%d", name, i.classAttribute().name(), (int) e.correct());
        System.out.println(namafile);
        System.out.println(e.toSummaryString(true));
        System.out.println(e.toMatrixString());
        saveModel(c, namafile + ".model");
    }
    
    private static void crossValidation(String name, Classifier c, Instances i) throws Exception {
        Evaluation e = new Evaluation(i);
        e.crossValidateModel(c, i, 10, new Random(1));
        String namafile = String.format("%s_%s_10-FOLD-CROSS-VALIDATION_%d", name, i.classAttribute().name(), (int) e.correct());
        System.out.println(namafile);
        System.out.println(e.toSummaryString(true));
        System.out.println(e.toMatrixString());
        saveModel(c, namafile + ".model");
    }
    
    private static void splitTest(String name, Instances i) throws Exception {
        i.randomize(new Random(0));
        int persen = 80;
        int trainSize = (int) Math.round(i.numInstances() * persen / 100);
        int testSize = i.numInstances() - trainSize;
        Instances iTrain = new Instances(i, 0, trainSize);
        iTrain.setClassIndex(i.classIndex());
        Instances iTest = new Instances(i, trainSize, testSize);
        iTest.setClassIndex(i.classIndex());
        Classifier c = new NaiveBayes();
        c.buildClassifier(iTrain);
        Evaluation e = new Evaluation(iTest);
        e.evaluateModel(c, iTest);
        String namafile = String.format("%s_%s_%d%%-SPLIT-TEST_%d", name, i.classAttribute().name(), persen, (int) e.correct());
        System.out.println(namafile);
        System.out.println(e.toSummaryString(true));
        System.out.println(e.toMatrixString());
        saveModel(c, namafile + ".model");
    }
    
    private static void experiment() throws Exception {
        Classifier c = new NaiveBayes();
        
        double max = -1; //variabel saat nilai maksimum
        double maxLearnRate = -1;
        int maxEpoch = -1;
        int maxNHidden = -1;
        int maxTrueAnswer = -1;
        Instances i = new ConverterUtils.DataSource("iris.arff").getDataSet(); //load data
        i.setClassIndex(i.numAttributes() - 1);
        Evaluation e = new Evaluation(i);
        System.out.println("Start " + new Date().toString());
        Discretize filter = new Discretize();
        filter.setInputFormat(i);
        i = Filter.useFilter(i, filter);
        
        c.buildClassifier(i);
        e.evaluateModel(c, i);
        System.out.println(e.toSummaryString(true));
        System.out.println(e.toMatrixString());
        System.out.println("Finish " + new Date().toString());
    }

    public static void main(String[] args) throws Exception {
        Thread.currentThread().setPriority(Thread.MAX_PRIORITY); //biar cepet
        //experiment(); System.exit(0); //komentari baris ini jika mau test manual
        Scanner s = new Scanner(System.in);
        System.out.print("File ARFF name : ");
        String name = s.next();
        Instances i = getInstances(s, name);
        Discretize filter = new Discretize();
        filter.setInputFormat(i);
        i = Filter.useFilter(i, filter);
        Classifier c = getClassifier(s, i);
        Evaluation e = getEvaluation(s, i, c, name);
        System.out.println(e.toClassDetailsString());
        System.out.println(e.toSummaryString(true));
        System.out.println(e.toMatrixString());
       
    }
}
