import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.Date;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class Tester {
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
                System.out.println("\nLearning rate; Number of hidden neuron; Number of epoch");
                System.out.print("Separated by space : ");
                c = new ANN(s.nextDouble(), s.nextInt(), s.nextInt(), i); //full training
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

    private static Evaluation getEvaluation(Scanner s, Instances i, Classifier c) throws Exception {
        System.out.println("\n1. Full training");
        System.out.println("2. 10-cross-fold validation");
        System.out.print("Choose : ");
        Evaluation e = new Evaluation(i);
        switch(s.nextInt()) {
            case 1 :
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

    private static void experiment(int idxClass) throws Exception {
        String modelname;
        Classifier c;
        Evaluation e;
        //current variable
        double learnRate;
        int nHidden;
        int epoch = 30000;
        int correct;
        //variabel saat nilai maksimum
        double maxLearnRate = -1;
        int maxNHidden = -1;
        int maxCorrect = -1;
        //load data
        Instances insTrain = new ConverterUtils.DataSource("strain.arff").getDataSet();
        Instances insTest = new ConverterUtils.DataSource("stest.arff").getDataSet();
        insTrain.setClassIndex(idxClass);
        insTest.setClassIndex(idxClass);
        //loop untuk mendapatkan parameter terbaik
        System.out.println("Learning rate, number of hidden neuron, number of epoch = correct answer");
        System.out.println("Start " + new Date().toString());
        for (learnRate = 0.005; learnRate <= 0.016; learnRate += 0.005) {
            for (nHidden = 10; nHidden <= 25; nHidden++) {
                //train dan evaluasi
                c = new ANN(learnRate, nHidden, epoch, insTest);
                c.buildClassifier(insTrain);
                e = new Evaluation(insTest);
                e.evaluateModel(c, insTest);
                correct = (int) e.correct();
                //cetak hasil
                if (correct >= maxCorrect) {
                    maxLearnRate = learnRate;
                    maxNHidden = nHidden;
                    maxCorrect = correct;
                    System.out.println(e.toSummaryString(true));
                    System.out.println(e.toMatrixString());
                    //save model terbaik
                    modelname = String.format("%d-%d-%.3f-%d-%d.model",
                        idxClass, maxCorrect, maxLearnRate, maxNHidden, epoch);
                    try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelname))) {
                        oos.writeObject(c);
                        oos.flush();
                    }
                }
            }
        }
        System.out.println("Finish " + new Date().toString());
        System.out.printf("Maksimum : %.3f %2d %5d = %3d dari %3d\n",
            maxLearnRate, maxNHidden, epoch, maxCorrect, insTest.numInstances());
    }

    public static void main(String[] args) throws Exception {
        Thread.currentThread().setPriority(Thread.MAX_PRIORITY); //biar cepet
        experiment(Integer.parseInt(args[0])); System.exit(0); //komentari baris ini jika mau test manual
        Scanner s = new Scanner(System.in);
        Instances i = getInstances(s);
        Classifier c = getClassifier(s, i);
        Evaluation e = getEvaluation(s, i, c);
        System.out.println("\n" + e.toSummaryString(true));
        System.out.println(e.toMatrixString());
        saveModel(s, c);
    }
}
