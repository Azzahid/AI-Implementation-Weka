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
    private double valThres = 0.5;
    private double learnRate = 0.1;
    private int nCol; //jumlah atribut = jumlah data + bias
    private int nRow;
    private int nHidden = 0;
    private int nOutput;
    private Node[] outNode;
    private Node[] hidNode;
    
    class Node {
        private final double[] weight;
        private double output;
        private final int nInput;
        
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
        
        public double count(double[] input) {
            //hitung sigma x dikali w
            double sigma = 0;
            for (int i = 0; i < nInput; i++) {
                sigma += input[i] * weight[i];
            }
            //masukkan sigma kedalam AF
            output = activationF(sigma);
            return output;
        }
        
        public void updateWeightOutput(double[] input, int target, int sign) {
            //ATTENTION: perihal nilai sign (0 atau 1)
            //misal <o1, o2, o3> = <0.6, 0.2, 0.8>
            //jika di-"step" disini maka <s1,s2,s3> = <1,0,1>
            //    (bisa melanggar aturan, seharusnya hanya boleh ada satu angka 1)
            //jika di-"step" diluar maka <s1,s2,s3> = <0,0,1>
            //    (pasti hanya ada satu angka 1 yaitu yang nilai o terbesar)
            //namun setelah beberapa eksperimen, lebih baik di-"step" disini
            sign = output > 0.5 ? 1 : 0; //hapus baris ini kalo mau "step"-an dari luar
            for (int i = 0; i < nInput; i++) {
                weight[i] += (learnRate * (target - sign) * input[i]);
            }
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
            Arrays.fill(sign, 0);
            //satu kali iterasi = hitung seluruh row
            for (int i = 0; i < nRow; i++) {
                //menghitung sign untuk setiap neuron output
                //proses menghitung sign nya di dalam prosedur Node.count()
                for (int j = 0; j < nOutput; j++) {
                    sign[j] = outNode[j].count(insNumeric[i]);
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
                //kumulatif half square error
                for (int j = 0; j < nOutput; j++) {
                    //ada neuron yang beda dari jawaban sebenarnya maka dianggap error
                    if ((int) sign[j] != target[i][j]) {
                        error += (Math.pow((sign[j] - target[i][j]), 2) / 2);
                        break; //tidak perlu cek neuron lainnya
                    }
                }
                //update bobot untuk setiap neuron
                for (int j = 0; j < nOutput; j++) {
                    outNode[j].updateWeightOutput(insNumeric[i], target[i][j], (int) sign[j]);
                }
                //satu row selesai
            }
            //satu iterasi selesai
            System.out.println("Iterasi " + iterasi + " Error " + error);
        } while(error > valThres);
        //cetak hasilnya yaitu bobot-bobot yang ada pada tiap neuron output
        printWeight(outNode, "output");
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
            outNode[i] = new Node(nHidden);
        }
        int iterasi = 0;
        double error;
        do {
            iterasi++;
            error = 0;
            //hitung UNDER CONSTRUCTION
            //backpropagation UNDER CONSTRUCTION
        } while(error > valThres);
        //cetak hasilnya yaitu bobot-bobot yang ada pada tiap neuron
        printWeight(hidNode, "hidden");
        printWeight(outNode, "output");
    }
    
    private void printWeight(Node[] node, String namaLayer) {
        System.out.println("\nBobot neuron " + namaLayer + ":");
        for (Node n : node) {
            double[] weight = n.getWeight();
            for (double d : weight) {
                System.out.printf("%.2f ", d);
            }
            System.out.println();
        }
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
    
    public static void main(String[] args) throws Exception {
        Instances i = new DataSource("iris.arff").getDataSet();
        i.setClassIndex(i.numAttributes() - 1); //kelas = atribut terakhir
        MultiLayerPerceptron mlp = new MultiLayerPerceptron();
        /* ATTENTION: ini sementara di komantarin dulu, pake nilai-nilai default aja */
        //System.out.print("Masukkan jumlah neuron pada hidden layer: ");
        //mlp.setHiddenLayers(new Scanner(System.in).nextInt());
        //mlp.setLearningRate(0.1);
        //mlp.setValidationThreshold(0.5);
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