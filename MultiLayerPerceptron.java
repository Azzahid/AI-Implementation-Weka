import java.util.Arrays;
import java.util.Scanner;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MultiLayerPerceptron extends AbstractClassifier {
    //nilai default valThres, learnRate, nHidden
    private double valThres = 0.05;
    private double learnRate = 0.1;
    private int nHidden = 2; //jumlah neuron pada layer hidden
    private int nOutput; //jumlah neuron pada layer output
    private int nCol; //jumlah atribut = jumlah data + bias
    private int nRow; //jumlah data set
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
                int idxMax = 0; //nilai sign yang paling besar
                for (int j = 0; j < nOutput; j++) {
                    outNode[j].countOutput(insNumeric[i]);
                    sign[j] = outNode[j].getOutput(); //penjelasan ada di dalam prosedurnya
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
        } while(error > valThres && iterasi < 10000);
        //cetak hasilnya yaitu bobot-bobot yang ada pada tiap neuron output
        System.out.printf("Iterasi %d error %.3f\n", iterasi, error);
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
                for (int j = 0; j < nHidden; j++) {
                    hidNode[j].countOutput(insNumeric[i]); //penjelasan ada di dalam prosedurnya
                    signHid[j + 1] = hidNode[j].getOutput();
                }
                //menghitung sign dan error untuk setiap neuron output
                for (int j = 0; j < nOutput; j++) {
                    outNode[j].countOutput(signHid); //penjelasan ada di dalam prosedurnya
                    outNode[j].countErrOut(target[i][j]); //penjelasan ada di dalam prosedurnya
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
        } while((error > valThres || error < -valThres) && iterasi < 5000);
        //cetak hasilnya yaitu bobot-bobot yang ada pada tiap neuron
        System.out.printf("Iterasi %d error %.3f\n", iterasi, error);
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
    
    private double[][] whateverToNumeric(Instances ins) {
        int kolom = ins.numAttributes();
        int baris = ins.numInstances();
        //filter menjadi numeric semua
        double[][] insNumeric = new double [baris][kolom + 1]; //kolom +1 untuk bias
        for (int i = 0; i < kolom; i++) {
            double[] numeric = ins.attributeToDoubleArray(i);
            for (int j = 0; j < baris; j++) {
                insNumeric[j][0] = 1; //bias ada di kolom pertama
                insNumeric[j][i + 1] = numeric[j];
            }
        }
        return insNumeric;
    }
    
    private int[][] buildTarget(double[][] insNumeric, int nOutput) {
        int baris = insNumeric.length;
        int kolom = insNumeric[0].length - 1; //-1 karena whateverToNumeric di +1
        //setiap row punya jawaban sebenarnya (target) yaitu tuple 1 & 0 sebanyak jenis kelasnya
        int[][] target = new int[baris][nOutput];
        if (nOutput == 1) { //jenis kelas boolean
            for (int i = 0; i < baris; i++) {
                target[i][0] = (int) insNumeric[i][kolom];
            }
        } else {
            for (int i = 0; i < baris; i++) {
                //dalam setiap row hanya ada satu angka 1, sisanya angka 0
                Arrays.fill(target[i], 0);
                target[i][(int) insNumeric[i][kolom]] = 1;
            }
        }
        return target;
    }
    
    @Override
    public void buildClassifier(Instances ins) throws Exception {
        nCol = ins.numAttributes();
        nRow = ins.numInstances();
        nOutput = ins.classAttribute().numValues();
        nOutput = nOutput == 2 ? 1 : nOutput; //jenis kelas boolean cukup 1 neuron output
        double[][] insNumeric = whateverToNumeric(ins);  //penjelasan ada di dalam prosedurnya
        int[][] target = buildTarget(insNumeric, nOutput); //penjelasan ada di dalam prosedurnya
        //single / multi perceptron
        if (nHidden == 0) {
            singlePerceptron(insNumeric, target);
        } else if(nHidden > 0) {
            singleHiddenLayer(insNumeric, target);
        }
    }
    
    public void testClassify(Instances ins) {
        int baris = ins.numInstances();
        int nKeluaran = ins.classAttribute().numValues();
        int benar = baris;
        nKeluaran = nKeluaran == 2 ? 1 : nKeluaran; //jenis kelas boolean cukup 1 neuron output
        double[][] insNumeric = whateverToNumeric(ins); //penjelasan ada di dalam prosedurnya
        int[][] target = buildTarget(insNumeric, nKeluaran); //penjelasan ada di dalam prosedurnya
        double[] signHid = new double[nHidden + 1]; //output dari layer hidden sebagai input layer output
        signHid[0] = 1; //+1 untuk bias yang ada di kolom pertama
        double[] signOut = new double[nKeluaran]; //hasil setelah AF lalu dijadikan 0 atau 1
        //testing seluruh baris data
        System.out.println("\nKlasifikasi tidak tepat : [classified] - [target]");
        for (int i = 0; i < baris; i++) {
            //menghitung sign untuk setiap neuron hidden
            for (int j = 0; j < nHidden; j++) { //single perceptron gabakal masuk sini
                hidNode[j].countOutput(insNumeric[i]); //penjelasan ada di dalam prosedurnya
                signHid[j + 1] = hidNode[j].getOutput();
            }
            //menghitung sign untuk setiap neuron output
            double[] input = nHidden == 0 ? insNumeric[i] : signHid;
            int idxMax = 0; //nilai sign yang paling besar
            for (int j = 0; j < nKeluaran; j++) {
                outNode[j].countOutput(input); //penjelasan ada di dalam prosedurnya
                signOut[j] = outNode[j].getOutput();
                /*WARNING: MLP tidak bisa begini karena sign nya bisa negatif!*/
                if (signOut[j] > signOut[idxMax]) {
                    idxMax = j;
                }
            }
            //ubah nilai sign menjadi 0 atau 1 (di-"step"-in)
            if (nKeluaran == 1) { //kelas boolean masuk sini
                signOut[0] = signOut[0] > 0.5 ? 1 : 0;
            } else {
                Arrays.fill(signOut, 0);
                signOut[idxMax] = 1;
            }
            //cek dengan jawaban sebenarnya
            for (int j = 0; j < nKeluaran; j++) {
                if ((int) signOut[j] != target[i][j]) { //target - sign != 0
                    System.out.println(Arrays.toString(signOut) + " - " + Arrays.toString(target[i]));
                    benar--;
                    break; //tidak perlu cek neuron lainnya
                }
            }
        }
        //print hasil
        System.out.println("Keakuratan : " + benar + " / " + baris);
    }
    
    public void crossValidation(Instances ins, int fold) throws Exception {
        for (int i = 0; i < fold; i++) {
            //folding
            Instances train = ins.trainCV(fold, i);
            train.setClassIndex(train.numAttributes() - 1); //kelas = atribut terakhir
            Instances test = ins.trainCV(fold, i);
            test.setClassIndex(test.numAttributes() - 1); //kelas = atribut terakhir
            //build and test
            System.out.printf("\n==========  FOLD-%d  ==========\n", i + 1);
            buildClassifier(train);
            testClassify(test);
        }
    }
    
    public static void main(String[] args) throws Exception {
        Instances i = new DataSource("iris.arff").getDataSet();
        i.setClassIndex(i.numAttributes() - 1); //kelas = atribut terakhir
        MultiLayerPerceptron mlp = new MultiLayerPerceptron();
        Scanner s = new Scanner(System.in);
        System.out.print("Masukkan jumlah neuron pada hidden layer: ");
        mlp.setHiddenLayers(s.nextInt());
        /*ATTENTION: sementara ini di jadiin komentar, pake nilai default dulu
        System.out.print("Masukkan konstanta learning rate        : ");
        mlp.setLearningRate(s.nextDouble());
        System.out.print("Masukkan konstanta validation threshold : ");
        mlp.setValidationThreshold(s.nextDouble());
        mlp.buildClassifier(i);
        mlp.testClassify(i); //full train test*/
        System.out.print("Masukkan jumlah fold : ");
        mlp.crossValidation(i, s.nextInt());
    }
}