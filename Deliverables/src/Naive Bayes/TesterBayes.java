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
import weka.filters.supervised.attribute.Discretize;

public class TesterBayes {
    private static Instances getInstances(Scanner s) throws Exception {
        System.out.print("File ARFF name : ");
        Instances i = new ConverterUtils.DataSource(s.next()).getDataSet();
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
        switch(s.nextInt()) {
            case 1 :
                System.out.print("\nLearning rate, number of hidden neuron, number of epoch : ");
                c = new NaiveBayes();
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

    private static Evaluation getEvaluation(Scanner s, Instances i, Classifier c) throws Exception {
        System.out.println("\n1. Full training");
        System.out.println("2. 10-cross-fold validation");
        System.out.print("Choose : ");
        Evaluation e = new Evaluation(i);
        Discretize filter = new Discretize();
        filter.setInputFormat(i);
        i = Filter.useFilter(i, filter);
        switch(s.nextInt()) {
            case 1 :
                c.buildClassifier(i);
                e.evaluateModel(c, i);
                break;
            case 2 :
                e.crossValidateModel(c, i, 10, new Random(1));
                break;
            default :
                System.exit(0);
        }
        return e;
    }

    private static void saveModel(Scanner s, Classifier c) throws Exception {
        System.out.println("1. Save model");
        System.out.println("2. Exit");
        System.out.print("Choose : ");
        switch(s.nextInt()) {
            case 1 :
                System.out.printf("\nFile MODEL name : ");
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(s.next()))) {
                    oos.writeObject(c);
                    oos.flush();
                }
                break;
            default :
                System.exit(0);
        }
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
        Instances i = getInstances(s);
        Classifier c = getClassifier(s, i);
        Evaluation e = getEvaluation(s, i, c);
        System.out.println(e.toClassDetailsString());
        System.out.println(e.toSummaryString(true));
        System.out.println(e.toMatrixString());
        saveModel(s, c);
    }
}
