import java.io.Serializable;
import java.util.Arrays;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class ANN extends AbstractClassifier implements Serializable {
    private final double learnRate;
    private final double valThres;
    private final int nHidden; //jumlah neuron pada layer hidden
    private int nOutput; //jumlah neuron pada layer output
    private int nCol; //jumlah atribut atau sama aja jumlah data + bias
    private int nRow; //jumlah instance
    private int idxClass; //index kolom posisi class
    private double[] avg;
    private double[] max; //nilai maksimum di setiap atribut
    private Neuron[] hidNode; //neuron pada layer hidden
    private Neuron[] outNode; //neuron pada layer output

    public ANN(double learnRate, double valThres, int nHidden) {
        this.learnRate = learnRate;
        this.valThres = valThres;
        this.nHidden = nHidden;
    }

    private void singleLayer(double[][] insNum, int[][] target) {
        outNode = new Neuron[nOutput]; //buat objek neuron-neuron pada layer output
        for (int i = 0; i < nOutput; i++) {
            outNode[i] = new Neuron(nCol);
        }
        double[] sign = new double[nOutput]; //hasil setelah AF lalu dijadikan 0 atau 1
        int iterasi = 0;
        double error;
        do {
            iterasi++;
            error = 0;
            //satu kali iterasi = hitung seluruh row
            for (int i = 0; i < nRow; i++) {
                //menghitung sign untuk setiap neuron output
                int idxMax = 0; //index neuron dengan nilai sign yang paling besar
                for (int j = 0; j < nOutput; j++) {
                    outNode[j].countSign(insNum[i]);
                    sign[j] = outNode[j].getSign();
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
                //hitung error kemudian update bobot
                boolean answer = true;
                for (int j = 0; j < nOutput; j++) {
                    outNode[j].countErrSingle(target[i][j], (int) sign[j]);
                    outNode[j].updateWeight(learnRate, insNum[i]);
                    //cek apakah tuple sign sesuai dengan tuple target
                    double err = outNode[j].getError();
                    if ((int) err != 0 && answer) {
                        error += (Math.pow(err, 2) / 2); //kumulatif half square error
                        answer = false; //tidak perlu cek neuron lainnya
                    }
                }
            }
        } while(error > valThres && iterasi < 5000);
        //cetak hasilnya yaitu bobot-bobot yang ada pada tiap neuron
        //System.out.printf("\nIterasi %d error %f\n", iterasi, error); //komentari baris ini jika experiment()
        //printWeight(); //komentari baris ini jika experiment()
    }

    private void dualLayer(double[][] insNum, int[][] target) {
        hidNode = new Neuron[nHidden];  //buat objek neuron-neuron pada layer hidden
        for (int i = 0; i < nHidden; i++) {
            hidNode[i] = new Neuron(nCol);
        }
        outNode = new Neuron[nOutput];  //buat objek neuron-neuron pada layer output
        for (int i = 0; i < nOutput; i++) {
            outNode[i] = new Neuron(nHidden + 1); //kolom +1 untuk bias
        }
        double[] signHid = new double[nHidden + 1]; //hasil setelah AF lalu masuk ke neuron output
        signHid[0] = 1; //bias di kolom pertama
        int iterasi = 0;
        double error;
        do {
            iterasi++;
            error = 0;
            //satu kali iterasi = hitung seluruh row
            for (int i = 0; i < nRow; i++) {
                //menghitung sign untuk setiap neuron hidden
                for (int j = 0; j < nHidden; j++) {
                    hidNode[j].countSign(insNum[i]);
                    signHid[j + 1] = hidNode[j].getSign();
                }
                //menghitung sign dan error untuk setiap neuron output
                for (int j = 0; j < nOutput; j++) {
                    outNode[j].countSign(signHid);
                    outNode[j].countErrOut(target[i][j]);
                }
                //menghitung error dan update bobot untuk setiap neuron hidden
                for (int j = 0; j < nHidden; j++) {
                    double sumErrXW = 0; //jumlah error x weight
                    for (Neuron n : outNode) {
                        sumErrXW += (n.getError() * n.getWeight()[j]);
                    }
                    hidNode[j].countErrHid(sumErrXW);
                    hidNode[j].updateWeight(learnRate, insNum[i]);
                }
                //update bobot untuk setiap neuron output
                for (Neuron n : outNode) {
                    n.updateWeight(learnRate, signHid);
                }
            }
            for (Neuron n : outNode) {
                error += (Math.pow(n.getError(), 2) / 2); //kumulatif half square error
            }
        } while(error > valThres && iterasi < 3000);
        //cetak hasilnya yaitu bobot-bobot yang ada pada tiap neuron
        //System.out.printf("\nIterasi %d error %f\n", iterasi, error); //komentari baris ini jika experiment()
        //printWeight(); //komentari baris ini jika experiment()
    }

    private double[][] whateverToNumeric(Instances ins) {
        double[][] insNum = new double [nRow][nCol + 1]; //kolom +1 untuk bias
        for (double[] i : insNum) {
            i[0] = 1; //bias di kolom pertama
        }
        int idxData = 1; //data dimulai dari kolom ke-2 (array idx = 1)
        for (int i = 0; i < nCol; i++) {
            double[] num = ins.attributeToDoubleArray(i);
            if (i == idxClass) {
                for (int j = 0; j < nRow; j++) {
                    insNum[j][nCol] = num[j]; //atribut di kolom terakhir
                }
            } else {
                for (int j = 0; j < nRow; j++) {
                    insNum[j][idxData] = num[j];
                }
                idxData++;
            }
        }
        return insNum;
    }

    private void zeroCNormalize(double[][] insNum) {
        double[] sum = new double[nCol]; //jumlah buat rata-rata
        Arrays.fill(sum, 0);
        max = new double[nCol]; //cari maksimum
        System.arraycopy(insNum[0], 0, max, 0, nCol);
        for (int i = 1; i < nRow; i++) {
            for (int j = 1; j < nCol; j++) { //bias tidak diperhitungkan
                sum[j] += insNum[i][j];
                if (insNum[i][j] > max[j]) {
                    max[j] = insNum[i][j];
                }
            }
        }
        avg = new double[nCol]; //rata-rata
        for (int i = 0; i < nCol; i++) {
            avg[i] = sum[i] / nRow;
        }
        for (int i = 0; i < nRow; i++) {
            for (int j = 1; j < nCol; j++) { //bias tidak diperhitungkan
                insNum[i][j] -= avg[j]; //zero center and normalize
                insNum[i][j] /= max[j];
            }
        }
        //FINAL STATE : insNum menjadi rentang -0.5 sampai 0.5
    }

    private int[][] buildTarget(double[][] insNum) {
        //setiap row punya target jawaban benar yaitu tuple 1 & 0 sebanyak jenis kelasnya
        int[][] target = new int[nRow][nOutput];
        if (nOutput == 1) { //kelas boolean masuk sini
            for (int i = 0; i < nRow; i++) {
                target[i][0] = (int) insNum[i][nCol];
            }
        } else { //multiple answer, dalam setiap row hanya ada satu angka 1 sisanya angka 0
            for (int i = 0; i < nRow; i++) {
                Arrays.fill(target[i], 0);
                target[i][(int) insNum[i][nCol]] = 1;
            }
        }
        return target;
    }

    @Override
    public double classifyInstance(Instance ins) throws Exception {
        double[] insNum = new double[nCol + 1]; //whateverToNumeric
        insNum[0] = 1; //bias di kolom pertama
        int idxData = 1; //data dimulai dari kolom ke-2 (array idx = 1)
        for (int i = 0; i < nCol; i++) {
            if (i == idxClass) {
                insNum[nCol] = ins.value(i);
            } else { //zero center and normalize
                insNum[idxData] = ins.value(i) - avg[idxData];
                insNum[idxData] /= max[idxData];
                idxData++;
            }
        }
        double[] inputToOutLayer = insNum;
        if (nHidden > 0) {
            double[] signHid = new double[nHidden + 1]; //hasil setelah AF lalu masuk ke neuron output
            signHid[0] = 1; //bias di kolom pertama
            for (int j = 0; j < nHidden; j++) { //menghitung sign untuk setiap neuron hidden
                hidNode[j].countSign(insNum);
                signHid[j + 1] = hidNode[j].getSign();
            }
            inputToOutLayer = signHid;
        }
        int idxMax = 0; //index neuron output dengan nilai sign yang paling besar
        double signMax = -1; //nilai sign output yang paling besar
        //menghitung sign untuk setiap neuron output sekaligus cari sign terbesar
        for (int j = 0; j < nOutput; j++) {
            outNode[j].countSign(inputToOutLayer);
            double sign = outNode[j].getSign();
            if (sign > signMax) {
                idxMax = j;
                signMax = sign;
            }
        }
        if (nOutput == 1) { //kelas boolean masuk sini
            return signMax > 0.5 ? 1 : 0;
        } else {
            //sign output tidak perlu diubah menjadi 0 atau 1, langsung return indeksnya
            return idxMax;
        }
    }

    @Override
    public void buildClassifier(Instances ins) throws Exception {
        nCol = ins.numAttributes();
        nRow = ins.numInstances();
        nOutput = ins.classAttribute().numValues();
        nOutput = nOutput == 2 ? 1 : nOutput; //jenis kelas boolean cukup 1 neuron output
        idxClass = ins.classIndex();
        double[][] insNum = whateverToNumeric(ins); //ubah ke numeric
        zeroCNormalize(insNum);
        int[][] target = buildTarget(insNum);
        if (nHidden == 0) {
            singleLayer(insNum, target);
        } else if(nHidden > 0) {
            dualLayer(insNum, target);
        }
    }

    private void printWeight() {
        /*
        tidak dikomentari, cuma style print doang
        */
        int nInputToOutLayer = nCol;
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
            nInputToOutLayer = nHidden + 1;
        }
        System.out.println("\nOutput Layer");
        for (int i = 0; i < nOutput; i++) {
            double[] weight = outNode[i].getWeight();
            System.out.printf("O%d -> bias(%.3f) ", i + 1, weight[0]);
            for (int j = 1; j < nInputToOutLayer; j++) {
                System.out.printf("w%d%d(%.3f) ", j, i + 1, weight[j]);
            }
            System.out.println();
        }
    }
}