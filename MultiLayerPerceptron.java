//import-import ini diambil dari MLP Weka yg asli, tapi banyak yg tidak di implemen disini
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.neural.LinearUnit;
import weka.classifiers.functions.neural.NeuralConnection;
import weka.classifiers.functions.neural.SigmoidUnit;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import weka.core.RevisionHandler;
import weka.core.WeightedInstancesHandler;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.RenameNominalValues;
import weka.filters.Filter;

public class MultiLayerPerceptron extends AbstractClassifier {
    //nilai default valThres, learnRate, nHidden
    private double valThres = 0.001;
    private double learnRate = 0.1;
    private int nCol; //jumlah atribut = jumlah data + bias
    private int nRow; //jumlah data set
    private int nHidden = 2; //jumlah neuron pada layer hidden
    private int nOutput; //jumlah neuron pada layer output
    private Node[] hidNode; //neuron pada layer hidden
    private Node[] outNode; //neuron pada layer output
    
    class Node {
        private double output; //nilai setelah keluar dari AF
        private double error;
        private final int nInput; //jumlah input yang masuk
        private final double[] weight;
        
        public Node(int nInput) {
            //tiap neuron punya bobot sebanyak inputnya (sudah termasuk bias)
            this.nInput = nInput;
            weight = new double[nInput];
            //untuk awal, semua bobot ditentukan bernilai 0
            Arrays.fill(weight, 0);
        }
        
        public Node(double[] predWeight) {
            //WARNING: constructor ini hanya digunakan saat load model
            //semua bobot sudah ditentukan nilainya
            weight = predWeight;
            nInput = weight.length;
        }
        
        private double activationF(double x) {
            //return x >= 0 ? 1 : 0; //step function dengan threshold = 0
            //return x >= 0 ? 1 : -1; //sign function
            return 1 / (1 + Math.pow(Math.E, -x)); //sigmoid function
        }
        
        public void countOutput(double[] input) {
            //hitung sigma x dikali w
            double sigma = 0;
            for (int i = 0; i < nInput; i++) {
                sigma += input[i] * weight[i];
            }
            //masukkan sigma kedalam AF
            output = activationF(sigma);
        }
        
        public void countErrSingle(int target, int sign) {
            //ATTENTION: perihal nilai sign (0 atau 1)
            //misal <o1, o2, o3> = <0.6, 0.2, 0.8>
            //jika di-"step" disini maka <s1,s2,s3> = <1,0,1>
            //    (bisa melanggar aturan, seharusnya hanya boleh ada satu angka 1)
            //jika di-"step" diluar maka <s1,s2,s3> = <0,0,1>
            //    (pasti hanya ada satu angka 1 yaitu yang nilai o terbesar)
            //sign = output > 0.5 ? 1 : 0; //hapus baris ini kalo mau "step"-an dari luar
            error = target - sign;
        }
        
        public void countErrOut(double target) {
            error = output * (1 - output) * (target - output); //ikutin rumus
        }
        
        public void countErrHid(double sum) {
            error = output * (1 - output) * sum; //ikutin rumus
        }
        
        public void updateWeight(double[] input) {
            //semua input bobotnya diatur ulang
            for (int i = 0; i < nInput; i++) {
                weight[i] += (learnRate * error * input[i]); //ikutin rumus
            }
        }
        
        public double getOutput() {
            return output;
        }
        
        public double getError() {
            return error;
        }
        
        public double[] getWeight() {
            return weight;
        }
    }
    
    public void setHiddenLayers(int h) {
        nHidden = h;
    }
    
    public void setLearningRate(double l) {
        learnRate = l;
    }

    public void setValidationThreshold(double t) {
        valThres = t;
    }
    
    private void singlePerceptron(double[][] insNumeric, int[][] target) {
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
                //proses menghitung sign nya di dalam prosedur Node.countOutput()
                for (int j = 0; j < nOutput; j++) {
                    outNode[j].countOutput(insNumeric[i]);
                    sign[j] = outNode[j].getOutput();
                }
                //dari semua neuron output, cari nilai sign yang paling besar
                int idxMax = 0;
                for (int j = 1; j < nOutput; j++) { //kelas boolean gabakal masuk sini
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
                    outNode[j].updateWeight(insNumeric[i]);
                    //kumulatif half square error
                    double err = outNode[j].getError();
                    if ((int) err != 0 && masihBenar) { //target - sign != 0
                        error += (Math.pow(err, 2) / 2);
                        masihBenar = false; //tidak perlu cek neuron lainnya
                    }
                }
                //satu row selesai
            }
            //satu iterasi selesai
            System.out.printf("Iterasi %d error %.3f\n", iterasi, error);
        } while(error > valThres && iterasi < 10000);
        //cetak hasilnya yaitu bobot-bobot yang ada pada tiap neuron output
        printModel();
    }
    
    private void singleHiddenLayer(double[][] insNumeric, int[][] target) {
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
                //proses menghitung sign nya di dalam prosedur Node.countOutput()
                for (int j = 0; j < nHidden; j++) {
                    hidNode[j].countOutput(insNumeric[i]);
                    signHid[j + 1] = hidNode[j].getOutput();
                }
                //menghitung sign dan error untuk setiap neuron output
                //proses menghitung sign nya di dalam prosedur Node.countOutput()
                //proses menghitung error nya di dalam prosedur Node.countErrOut()
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
                    n.updateWeight(insNumeric[i]);
                }
                //update bobot untuk setiap neuron output
                for (Node n : outNode) {
                    n.updateWeight(signHid);
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
        printModel();
    }
    
    private void printModel() {
        int input = nCol; //input hidden dan output bisa beda
        System.out.println("\n=== Classifier model ===");
        System.out.println("\nFeed Forward Neural Network");
        System.out.println("---------------------------");
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
        System.out.println("\nValidation Threshold : " + valThres);
        System.out.println("Learning Rate : " + learnRate);
        System.out.println("Number of Hidden Neuron : " + nHidden);
        System.out.println("Number of Output Neuron : " + nOutput);
    }
    
    @Override
    public void buildClassifier(Instances ins) throws Exception {
        nCol = ins.numAttributes();
        nRow = ins.numInstances();
        nOutput = ins.classAttribute().numValues();
        //filter menjadi numeric semua
        double[][] insNumeric = new double [nRow][nCol + 1]; //kolom +1 untuk bias
        for (int i = 0; i < nCol; i++) {
            double[] numeric = ins.attributeToDoubleArray(i);
            for (int j = 0; j < nRow; j++) {
                insNumeric[j][0] = 1; //bias ada di kolom pertama
                insNumeric[j][i + 1] = numeric[j];
            }
        }
        //setiap row punya jawaban sebenarnya (target) yaitu tuple 1 & 0 sebanyak jenis kelasnya
        int[][] target = new int[nRow][nOutput];
        if (nOutput == 2) { //jenis kelas cuma ada 2 (misal: boolean TRUE/FALSE), cukup 1 neuron output
            nOutput = 1;
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
        //single / multi perceptron
        if (nHidden == 0) {
            singlePerceptron(insNumeric, target);
        } else if(nHidden > 0) {
            singleHiddenLayer(insNumeric, target);
        }
    }
    
    public void test(Instances dataTes) {
        
    }
    
    public void inputTest() {
        
    }
    
    public static void main(String[] args) throws Exception {
        Instances i = new DataSource("iris.arff").getDataSet();
        i.setClassIndex(i.numAttributes() - 1); //kelas = atribut terakhir
        MultiLayerPerceptron mlp = new MultiLayerPerceptron();
        //ATTENTION: sementara ini di jadiin komentar, pake nilai default dulu
        Scanner s = new Scanner(System.in);
        System.out.print("Masukkan jumlah neuron pada hidden layer: ");
        mlp.setHiddenLayers(s.nextInt());
        /*System.out.print("Masukkan konstanta learning rate        : ");
        mlp.setLearningRate(s.nextDouble());
        System.out.print("Masukkan konstanta validation threshold : ");
        mlp.setValidationThreshold(s.nextDouble());*/
        mlp.buildClassifier(i);
    }
}

//<editor-fold defaultstate="collapsed" desc="Tidak Di Implemen">
/*
BELUM / TIDAK DI IMPLEMENTASI

    public void setDecay(boolean d) {
        
    }
    
    public boolean getDecay() {
        return false;
    }
    
    public void setReset(boolean r) {
        
    }
    
    public boolean getReset() {
        return false;
    }
    
    public void setNormalizeNumericClass(boolean c) {
        
    }

    public boolean getNormalizeNumericClass() {
        return false;
    }

    public void setNormalizeAttributes(boolean a) {
        
    }

    public boolean getNormalizeAttributes() {
        return false;
    }

    public void setNominalToBinaryFilter(boolean f) {
        
    }

    public boolean getNominalToBinaryFilter() {
        return false;
    }

    public void setSeed(int l) {
        
    }

    public int getSeed() {
        return 0;
    }

    public void setMomentum(double m) {
        
    }

    public double getMomentum() {
        return 0;
    }

    public void setAutoBuild(boolean a) {
        
    }

    public boolean getAutoBuild() {
        return false;
    }

    public void setGUI(boolean a) {
        
    }

    public boolean getGUI() {
        return false;
    }

    public void setValidationSetSize(int a) {
        
    }

    public int getValidationSetSize() {
        return 0;
    }

    public void setTrainingTime(int n) {
        
    }

    public int getTrainingTime() {
        return 0;
    }
    
    public synchronized void blocker(boolean tf) {
        
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }

    @Override
    public double[] distributionForInstance(Instance i) throws Exception {
        return null;
    }

    @Override
    public Enumeration<Option> listOptions() {
        return null;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        
    }

    @Override
    public String[] getOptions() {
        return null;
    }

    @Override
    public String toString() {
        return null;
    }

    public String globalInfo() {
        return null;
    }

    @Override
    public String getRevision() {
        return null;
    }

*/
//</editor-fold>