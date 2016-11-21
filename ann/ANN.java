import java.io.Serializable;
import java.util.Arrays;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class ANN extends AbstractClassifier implements Serializable {
    private final double learnRate;
    private final int nHidden; //jumlah neuron pada layer hidden
    private final int maxEpoch; //learning berhenti berdasarkan epoch
    private int nOutput; //jumlah neuron pada layer output
    private int nCol; //jumlah atribut atau sama aja jumlah data + bias
    private int nRow; //jumlah instance
    private int idxClass; //index kolom posisi class
    private double[] avg;
    private double[] max; //nilai maksimum di setiap atribut
    private Neuron[] hidNeuron; //neuron pada layer hidden
    private Neuron[] outNeuron; //neuron pada layer output

    public ANN(double learnRate, int nHidden, int maxEpoch) {
        this.learnRate = learnRate;
        this.nHidden = nHidden;
        this.maxEpoch = maxEpoch;
    }

    private void singleLayer(double[][] insNum, int[][] target) {
        outNeuron = new Neuron[nOutput]; //buat objek neuron-neuron pada layer output
        for (int i = 0; i < nOutput; i++) {
            outNeuron[i] = new Neuron(nCol);
        }
        int epoch = 0;
        do {
            epoch++;
            //satu kali epoch = hitung seluruh row
            for (int i = 0; i < nRow; i++) {
                //satu kali hitung row = hitung seluruh neuron output
                for (int j = 0; j < nOutput; j++) {
                    outNeuron[j].countSign(insNum[i]); //hitung sign
                    outNeuron[j].countErrSingle(target[i][j]); //hitung error
                    outNeuron[j].updateWeight(learnRate, insNum[i]); //update bobot
                }
            }
        } while(epoch < maxEpoch);
        //printWeight(); //cetak hasil, komentari baris ini jika experiment
    }

    private void dualLayer(double[][] insNum, int[][] target) {
        hidNeuron = new Neuron[nHidden];  //buat objek neuron-neuron pada layer hidden
        for (int i = 0; i < nHidden; i++) {
            hidNeuron[i] = new Neuron(nCol);
        }
        outNeuron = new Neuron[nOutput];  //buat objek neuron-neuron pada layer output
        for (int i = 0; i < nOutput; i++) {
            outNeuron[i] = new Neuron(nHidden + 1); //kolom +1 untuk bias
        }
        double[] signHid = new double[nHidden + 1]; //hasil setelah AF lalu masuk ke neuron output
        signHid[0] = 1; //bias di kolom pertama
        int epoch = 0;
        do {
            epoch++;
            //satu kali epoch = hitung seluruh row
            for (int i = 0; i < nRow; i++) {
                //menghitung sign untuk setiap neuron hidden
                for (int j = 0; j < nHidden; j++) {
                    hidNeuron[j].countSign(insNum[i]);
                    signHid[j + 1] = hidNeuron[j].getSign();
                }
                //menghitung sign dan error untuk setiap neuron output
                for (int j = 0; j < nOutput; j++) {
                    outNeuron[j].countSign(signHid);
                    outNeuron[j].countErrOut(target[i][j]);
                }
                //menghitung error dan update bobot untuk setiap neuron hidden
                for (int j = 0; j < nHidden; j++) {
                    double sumErrXW = 0; //jumlah error x weight
                    for (Neuron n : outNeuron) {
                        sumErrXW += (n.getError() * n.getWeight()[j]);
                    }
                    hidNeuron[j].countErrHid(sumErrXW);
                    hidNeuron[j].updateWeight(learnRate, insNum[i]);
                }
                //update bobot untuk setiap neuron output
                for (Neuron n : outNeuron) {
                    n.updateWeight(learnRate, signHid);
                }
            }
        } while(epoch < maxEpoch);
        //printWeight(); //cetak hasil, komentari baris ini jika experiment
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
                    insNum[j][nCol] = num[j]; //atribut kelas di kolom terakhir
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
        //cari maksimum untuk setiap atribut
        max = new double[nCol];
        System.arraycopy(insNum[0], 0, max, 0, nCol);
        for (double[] i : insNum) {
            for (int j = 1; j < nCol; j++) { //bias tidak diperhitungkan
                sum[j] += i[j];
                if (i[j] > max[j]) {
                    max[j] = i[j];
                }
            }
        }
        //hitung rata-rata untuk setiap atribut
        avg = new double[nCol];
        for (int i = 0; i < nCol; i++) {
            avg[i] = sum[i] / nRow;
            max[i] -= avg[i];
        }
        //zero center lalu normalize (ubah insNum menjadi rentang -1 sampai 1)
        for (int i = 0; i < nRow; i++) {
            for (int j = 1; j < nCol; j++) { //bias tidak diubah
                insNum[i][j] -= avg[j];
                insNum[i][j] /= max[j];
            }
        }
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
            if (i == idxClass) { //atribut kelas di kolom terakhir
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
                hidNeuron[j].countSign(insNum);
                signHid[j + 1] = hidNeuron[j].getSign();
            }
            inputToOutLayer = signHid;
        }
        int idxMax = 0; //index neuron output dengan nilai sign yang paling besar
        double signMax = -1; //nilai sign output yang paling besar
        //menghitung sign untuk setiap neuron output sekaligus cari sign terbesar
        for (int j = 0; j < nOutput; j++) {
            outNeuron[j].countSign(inputToOutLayer);
            double sign = outNeuron[j].getSign();
            if (sign > signMax) {
                idxMax = j;
                signMax = sign;
            }
        }
        if (nOutput == 1) { //kelas boolean masuk sini
            return signMax > 0.5 ? 1 : 0;
        } else { //sign output tidak perlu diubah menjadi 0 atau 1
            return idxMax;
        }
    }

    @Override
    public void buildClassifier(Instances ins) throws Exception {
        nCol = ins.numAttributes();
        nRow = ins.numInstances();
        idxClass = ins.classIndex();
        nOutput = ins.numDistinctValues(idxClass);
        nOutput = nOutput == 2 ? 1 : nOutput; //jenis kelas boolean cukup 1 neuron output
        double[][] insNum = whateverToNumeric(ins);
        zeroCNormalize(insNum);
        int[][] target = buildTarget(insNum);
        if (nHidden == 0) {
            singleLayer(insNum, target);
        } else if(nHidden > 0) {
            dualLayer(insNum, target);
        }
    }

    private void printWeight() {
        //cetak model (semua bobot yang ada pada semua neuron)
        int nInputToOutLayer = nCol;
        if (nHidden > 0) {
            System.out.println("\nHidden Layer");
            for (int i = 0; i < nHidden; i++) {
                double[] weight = hidNeuron[i].getWeight();
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
            double[] weight = outNeuron[i].getWeight();
            System.out.printf("O%d -> bias(%.3f) ", i + 1, weight[0]);
            for (int j = 1; j < nInputToOutLayer; j++) {
                System.out.printf("w%d%d(%.3f) ", j, i + 1, weight[j]);
            }
            System.out.println();
        }
    }
}