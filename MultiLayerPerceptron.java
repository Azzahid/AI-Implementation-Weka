import java.util.Arrays;
import java.util.Scanner;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class MultiLayerPerceptron extends AbstractClassifier {
    //nilai default valThres, learnRate, nHidden
    private double learnRate = 0.1;
    private double valThres = 0.05;
    private int nHidden = 2; //jumlah neuron pada layer hidden
    private int nOutput; //jumlah neuron pada layer output
    private int nCol; //jumlah atribut = jumlah data + bias
    private int nRow; //jumlah data set
    private Node[] hidNode; //neuron pada layer hidden
    private Node[] outNode; //neuron pada layer output
    
    public MultiLayerPerceptron(double learnRate, double valThres, int nHidden) {
        this.learnRate = learnRate;
        this.valThres = valThres;
        this.nHidden = nHidden;
    }
    
    public void setLearningRate(double l) {
        learnRate = l;
    }

    public void setValidationThreshold(double t) {
        valThres = t;
    }
    
    public void setHiddenLayers(int h) {
        nHidden = h;
    }
    
    private void singleLayer(double[][] insNumeric, int[][] target) {
        //neuron-neuron di layer output
        outNode = new Node[nOutput];
        for (int i = 0; i < nOutput; i++) {
            outNode[i] = new Node(nCol);
        }
        double[] sign = new double[nOutput]; //hasil setelah AF lalu dijadikan 0 atau 1
        double error;
        int iterasi = 0;
        do {
            iterasi++;
            error = 0;
            //satu kali iterasi = hitung seluruh row
            for (int i = 0; i < nRow; i++) {
                //menghitung sign untuk setiap neuron output
                int idxMax = 0; //nilai sign yang paling besar
                for (int j = 0; j < nOutput; j++) {
                    outNode[j].countOutput(insNumeric[i]);
                    sign[j] = outNode[j].getOutput();
                    if (sign[j] > sign[idxMax]) {
                        idxMax = j;
                    }
                }
                //ubah nilai sign menjadi 0 atau 1 (di-"step"-in)
                if (nOutput == 1) { //kelas boolean masuk sini
                    sign[0] = sign[0] > 0.5 ? 1 : 0;
                } else {
                    Arrays.fill(sign, 0);
                    sign[idxMax] = 1;
                }
                boolean masihBenar = true;
                for (int j = 0; j < nOutput; j++) {
                    //update bobot untuk setiap neuron
                    outNode[j].countErrSingle(target[i][j], (int) sign[j]);
                    outNode[j].updateWeight(learnRate, insNumeric[i]);
                    //kumulatif half square error
                    double err = outNode[j].getError();
                    if ((int) err != 0 && masihBenar) { //target - sign != 0
                        error += (Math.pow(err, 2) / 2);
                        masihBenar = false; //tidak perlu cek neuron lainnya
                    }
                }
            }
            System.out.printf("Iterasi %d error %.3f\n", iterasi, error);
        } while(error > valThres && iterasi < 10000);
        //cetak hasilnya yaitu bobot-bobot yang ada pada tiap neuron output
        printWeight();
    }
    
    private void dualLayer(double[][] insNumeric, int[][] target) {
        //neuron-neuron di layer hidden
        hidNode = new Node[nHidden];
        for (int i = 0; i < nHidden; i++) {
            hidNode[i] = new Node(nCol);
        }
        //neuron-neuron di layer output
        outNode = new Node[nOutput];
        for (int i = 0; i < nOutput; i++) {
            outNode[i] = new Node(nHidden + 1); //kolom +1 untuk bias
        }
        int iterasi = 0;
        double error;
        double[] signHid = new double[nHidden + 1]; //output dari layer hidden sebagai input layer output
        signHid[0] = 1;
        do {
            iterasi++;
            error = 0;
            //satu kali iterasi = hitung seluruh row
            for (int i = 0; i < nRow; i++) {
                //menghitung sign untuk setiap neuron hidden
                for (int j = 0; j < nHidden; j++) {
                    hidNode[j].countOutput(insNumeric[i]);
                    signHid[j + 1] = hidNode[j].getOutput();
                }
                //menghitung sign dan error untuk setiap neuron output
                for (int j = 0; j < nOutput; j++) {
                    outNode[j].countOutput(signHid);
                    outNode[j].countErrOut(target[i][j]);
                }
                //menghitung error untuk setiap neuron hidden
                for (int j = 0; j < nHidden; j++) {
                    //jumlahin error x weight
                    double sumErrXW = 0;
                    for (Node n : outNode) {
                        sumErrXW += (n.getError() * n.getWeight()[j]);
                    }
                    //proses menghitung error nya di dalam prosedur Node.countErrHid()
                    hidNode[j].countErrHid(sumErrXW);
                }
                //update bobot untuk setiap neuron hidden
                for (Node n : hidNode) {
                    n.updateWeight(learnRate, insNumeric[i]);
                }
                //update bobot untuk setiap neuron output
                for (Node n : outNode) {
                    n.updateWeight(learnRate, signHid);
                }
                //satu row selesai
            }
            //satu iterasi selesai
            for (Node n : outNode) { //ini ngarang, jumlahin semua error supaya bisa iterasi lagi
                error += n.getError();
            }
            System.out.printf("Iterasi %d error %.3f\n", iterasi, error);
        } while((error > valThres || error < -valThres) && iterasi < 5000);
        //cetak hasilnya yaitu bobot-bobot yang ada pada tiap neuron
        printWeight();
    }
    
    private double[][] whateverToNumeric(Instances ins) {
        //filter menjadi numeric semua
        double[][] insNumeric = new double [nRow][nCol + 1]; //kolom +1 untuk bias
        for (int i = 0; i < nCol; i++) {
            double[] numeric = ins.attributeToDoubleArray(i);
            for (int j = 0; j < nRow; j++) {
                insNumeric[j][0] = 1; //bias ada di kolom pertama
                insNumeric[j][i + 1] = numeric[j];
            }
        }
        return insNumeric;
    }
    
    private int[][] buildTarget(double[][] insNumeric) {
        //setiap row punya jawaban sebenarnya (target) yaitu tuple 1 & 0 sebanyak jenis kelasnya
        int[][] target = new int[nRow][nOutput];
        if (nOutput == 1) { //jenis kelas boolean
            for (int i = 0; i < nRow; i++) {
                target[i][0] = (int) insNumeric[i][nCol];
            }
        } else {
            for (int i = 0; i < nRow; i++) {
                //dalam setiap row hanya ada satu angka 1, sisanya angka 0
                Arrays.fill(target[i], 0);
                target[i][(int) insNumeric[i][nCol]] = 1;
            }
        }
        return target;
    }
    
    @Override
    public double classifyInstance(Instance ins) throws Exception {
        //whateverToNumeric
        double[] insNumeric = new double[nCol + 1];
        insNumeric[0] = 1;
        for (int i = 0; i < nCol; i++) {
            insNumeric[i + 1] = ins.value(i);
        }
        double[] input = insNumeric;
        if (nHidden > 0) {
            //output dari layer hidden sebagai input layer output
            double[] signHid = new double[nHidden + 1];
            signHid[0] = 1; //+1 untuk bias yang ada di kolom pertama
            //menghitung sign untuk setiap neuron hidden
            for (int j = 0; j < nHidden; j++) { //single perceptron gabakal masuk sini
                hidNode[j].countOutput(insNumeric);
                signHid[j + 1] = hidNode[j].getOutput();
            }
            input = signHid;
        }
        //menghitung sign untuk setiap neuron output
        int idxMax = 0; //nilai sign yang paling besar
        double signOut = -1; //paling kecil karena gak mungkin ada sign negatif
        //hasil setelah AF lalu dijadikan 0 atau 1
        int j = 0;
        do {
            outNode[j].countOutput(input);
            double newSign = outNode[j].getOutput();
            if (newSign > signOut) {
                idxMax = j;
                signOut = newSign;
            }
            j++;
        } while(j < nOutput);
        //ubah nilai sign menjadi 0 atau 1 (di-"step"-in)
        if (nOutput == 1) { //kelas boolean masuk sini
            return signOut > 0.5 ? 1 : 0;
        } else {
            return idxMax;
        }
    }
    
    @Override
    public void buildClassifier(Instances ins) throws Exception {
        nCol = ins.numAttributes();
        nRow = ins.numInstances();
        nOutput = ins.classAttribute().numValues();
        nOutput = nOutput == 2 ? 1 : nOutput; //jenis kelas boolean cukup 1 neuron output
        double[][] insNumeric = whateverToNumeric(ins);
        int[][] target = buildTarget(insNumeric);
        //single / multi perceptron
        if (nHidden == 0) {
            singleLayer(insNumeric, target);
        } else if(nHidden > 0) {
            dualLayer(insNumeric, target);
        }
    }
    
    private void printWeight() {
        int input = nCol; //input hidden dan output bisa beda
        if (nHidden > 0) {
            System.out.println("\nHidden Layer");
            for (int i = 0; i < nHidden; i++) {
                double[] weight = hidNode[i].getWeight();
                System.out.printf("H%d -> bias(%.3f) ", i + 1, weight[0]);
                for (int j = 1; j < nCol; j++) {
                    System.out.printf("w%d%d(%.3f) ", j, i + 1, weight[j]);
                }
                System.out.println();
            }
            input = nHidden + 1;
        }
        System.out.println("\nOutput Layer");
        for (int i = 0; i < nOutput; i++) {
            double[] weight = outNode[i].getWeight();
            System.out.printf("O%d -> bias(%.3f) ", i + 1, weight[0]);
            for (int j = 1; j < input; j++) {
                System.out.printf("w%d%d(%.3f) ", j, i + 1, weight[j]);
            }
            System.out.println();
        }
    }
    
    public static void main(String[] args) throws Exception {
        Instances i = new DataSource("Team.arff").getDataSet();
        i.setClassIndex(i.numAttributes() - 1); //kelas = atribut terakhir
        //normalisasi
        Normalize filter = new Normalize();
        filter.setInputFormat(i);
        i = Filter.useFilter(i, filter);
        //train
        Scanner s = new Scanner(System.in);
        System.out.print("Learning rate, validation threshold, number of hidden neuron : ");
        Classifier mlp = new MultiLayerPerceptron(s.nextDouble(), s.nextDouble(), s.nextInt());
        mlp.buildClassifier(i);
        Evaluation eval = new Evaluation(i);
        eval.evaluateModel(mlp, i);
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }
}