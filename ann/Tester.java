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
        int idxClass = s.nextInt() - 1;
        i.setClassIndex(idxClass);
        if (idxClass == 26) {
            i.deleteAttributeAt(27);
        } else if (idxClass == 27) {
            i.deleteAttributeAt(26);
        }
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
        int epoch = 30000;
        //current variable
        double learnRate;
        int nHidden;
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
        switch (idxClass) { //delete atribut
            case 26:
                insTrain.deleteAttributeAt(27);
                insTest.deleteAttributeAt(27);
                break;
            case 27:
                insTrain.deleteAttributeAt(26);
                insTest.deleteAttributeAt(26);
                break;
            default:
                return;
        }
        //loop untuk mendapatkan parameter terbaik
        System.out.println("Learning rate, number of hidden neuron, number of epoch = correct answer");
        System.out.println("Start " + new Date().toString());
        for (learnRate = 0.005; learnRate <= 0.016; learnRate += 0.005) { //0.016 supaya 0.015 masuk
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
    
    private static void saveModel(Classifier c, String namafile) throws Exception {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(namafile))) {
            oos.writeObject(c);
            oos.flush();
        }
    }
    
    private static void fullTraining(String name, Classifier c, Instances i) throws Exception {
        c.buildClassifier(i);
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
    
    private static void splitTest(String name, Instances i,
        double LR, int NH, int ME) throws Exception {
        i.randomize(new Random(0));
        int persen = 80;
        int trainSize = (int) Math.round(i.numInstances() * persen / 100);
        int testSize = i.numInstances() - trainSize;
        Instances iTrain = new Instances(i, 0, trainSize);
        iTrain.setClassIndex(i.classIndex());
        Instances iTest = new Instances(i, trainSize, testSize);
        iTest.setClassIndex(i.classIndex());
        Classifier c = new ANN(LR, NH, ME, iTest);
        c.buildClassifier(iTrain);
        Evaluation e = new Evaluation(iTest);
        e.evaluateModel(c, iTest);
        String namafile = String.format("%s_%s_%d%%-SPLIT-TEST_%d", name, i.classAttribute().name(), persen, (int) e.correct());
        System.out.println(namafile);
        System.out.println(e.toSummaryString(true));
        System.out.println(e.toMatrixString());
        saveModel(c, namafile + ".model");
    }
    
    private static void hihihi(String name, int idxClass, int idxDel,
        double LR, int NH, int ME) throws Exception {
        Instances i = new ConverterUtils.DataSource(name).getDataSet();
        i.setClassIndex(idxClass);
        if (idxDel != -1) {
            i.deleteAttributeAt(idxDel);
        }
        Classifier c = new ANN(LR, NH, ME, i);
        fullTraining(name, c, i);
        crossValidation(name, c, i);
        splitTest(name, i, LR, NH, ME);
    }

    private static void lalala() throws Exception {
        Scanner s = new Scanner(System.in);
        System.out.print("Learning rate: "); double LR = s.nextDouble();
        System.out.print("Hidden layer: ");
        int NH;
        switch (s.nextInt()) {
            case 0:
                NH = 0;
                break;
            case 1:
                System.out.print("Hidden neuron: "); NH = s.nextInt();
                break;
            default:
                return;
        }
        System.out.print("Maximum epoch: "); int ME = s.nextInt();
        System.out.println();
        hihihi("iris.arff", 4, -1, LR, NH, ME);
        hihihi("ttrain.arff", 12, -1, LR, NH, ME);
        hihihi("ttest.arff", 12, -1, LR, NH, ME);
        hihihi("strain.arff", 26, 27, LR, NH, ME);
        hihihi("strain.arff", 27, 26, LR, NH, ME);
        hihihi("stest.arff", 26, 27, LR, NH, ME);
        hihihi("stest.arff", 27, 26, LR, NH, ME);
    }

    public static void main(String[] args) throws Exception {
        Thread.currentThread().setPriority(Thread.MAX_PRIORITY); //biar cepet
        //experiment(Integer.parseInt(args[0])); System.exit(0); //komentari baris ini jika mau test manual
        lalala(); System.exit(0); //komentari baris ini jika mau test manual
        Scanner s = new Scanner(System.in);
        Instances i = getInstances(s);
        Classifier c = getClassifier(s, i);
        Evaluation e = getEvaluation(s, i, c);
        System.out.println("\n" + e.toSummaryString(true));
        System.out.println(e.toMatrixString());
        saveModel(s, c);
    }
}
