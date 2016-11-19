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

    private void singleLayer(double[][] insNumeric, int[][] target) {
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
                    outNode[j].countSign(insNumeric[i]);
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
                boolean answer = true;
                for (int j = 0; j < nOutput; j++) {
                    //hitung error kemudian update bobot
                    outNode[j].countErrSingle(target[i][j], (int) sign[j]);
                    outNode[j].updateWeight(learnRate, insNumeric[i]);
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
        //System.out.printf("\nIterasi %d error %.3f\n", iterasi, error); //komentari baris ini jika experiment()
        //printWeight(); //komentari baris ini jika experiment()
    }

    private void dualLayer(double[][] insNumeric, int[][] target) {
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
                    hidNode[j].countSign(insNumeric[i]);
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
                    hidNode[j].updateWeight(learnRate, insNumeric[i]);
                }
                //update bobot untuk setiap neuron output
                for (Neuron n : outNode) {
                    n.updateWeight(learnRate, signHid);
                }
            }
            for (Neuron n : outNode) { //ini ngarang, jumlahin semua error untuk menentukan iterasi selanjutnya
                error += n.getError();
            }
        } while((error > valThres || error < -valThres) && iterasi < 3000);
        //cetak hasilnya yaitu bobot-bobot yang ada pada tiap neuron
        //System.out.printf("\nIterasi %d error %.3f\n", iterasi, error); //komentari baris ini jika experiment()
        //printWeight(); //komentari baris ini jika experiment()
    }

    private double[][] whateverToNumeric(Instances ins) {
        /*
        UNDER CONSTRUCTION
        memindahkan class dimanapun posisi atribut awalnya menjadi di kolom terakhir
        */
        double[][] insNumeric = new double [nRow][nCol + 1]; //kolom +1 untuk bias
        for (int i = 0; i < nCol; i++) { //ASUMSI class di atribut terakhir
            double[] numeric = ins.attributeToDoubleArray(i);
            for (int j = 0; j < nRow; j++) {
                insNumeric[j][0] = 1; //bias di kolom pertama
                insNumeric[j][i + 1] = numeric[j];
            }
        }
        return insNumeric;
    }

    private double[][] normalize(double[][] insNumeric) {
        double[][] insNormalize = new double[nRow][nCol + 1]; //duplicate
        for (int i = 0; i < nRow; i++) {
            System.arraycopy(insNumeric[i], 0, insNormalize[i], 0, nCol + 1);
        }
        double[] sum = new double[nCol];
        Arrays.fill(sum, 0);
        max = new double[nCol]; //cari maksimum
        System.arraycopy(insNumeric[0], 0, max, 0, nCol);
        for (int i = 1; i < nRow; i++) {
            for (int j = 1; j < nCol; j++) {
                sum[j] += insNumeric[i][j];
                if (insNumeric[i][j] > max[j]) {
                    max[j] = insNumeric[i][j];
                }
            }
        }
        avg = new double[nCol]; //cari rata-rata
        for (int i = 0; i < nCol; i++) {
            avg[i] = sum[i] / nRow;
        }
        for (int i = 0; i < nRow; i++) { //zero center and normalize
            for (int j = 1; j < nCol; j++) {
                insNormalize[i][j] -= avg[j];
                insNormalize[i][j] /= max[j];
            }
        }
        return insNormalize;
    }

    private int[][] buildTarget(double[][] insNumeric) {
        //setiap row punya target jawaban benar yaitu tuple 1 & 0 sebanyak jenis kelasnya
        int[][] target = new int[nRow][nOutput];
        if (nOutput == 1) { //kelas boolean masuk sini
            for (int i = 0; i < nRow; i++) {
                target[i][0] = (int) insNumeric[i][nCol];
            }
        } else { //multiple answer, dalam setiap row hanya ada satu angka 1 sisanya angka 0
            for (int i = 0; i < nRow; i++) {
                Arrays.fill(target[i], 0);
                target[i][(int) insNumeric[i][nCol]] = 1;
            }
        }
        return target;
    }

    @Override
    public double classifyInstance(Instance ins) throws Exception {
        /*
        UNDER CONSTRUCTION
        memindahkan class dimanapun posisi atribut awalnya menjadi di kolom terakhir
        */
        //whateverToNumeric() sekaligus normalize()
        double[] insNumeric = new double[nCol + 1];
        insNumeric[0] = 1; //bias di kolom pertama
        for (int i = 0; i < nCol - 1; i++) { //ASUMSI class di atribut terakhir
            insNumeric[i + 1] = ins.value(i) - avg[i + 1];
            insNumeric[i + 1] /= max[i + 1];
        }
        insNumeric[nCol] = ins.value(idxClass - 1);
        double[] inputToOutLayer = insNumeric;
        if (nHidden > 0) {
            double[] signHid = new double[nHidden + 1]; //hasil setelah AF lalu masuk ke neuron output
            signHid[0] = 1; //bias di kolom pertama
            for (int j = 0; j < nHidden; j++) { //menghitung sign untuk setiap neuron hidden
                hidNode[j].countSign(insNumeric);
                signHid[j + 1] = hidNode[j].getSign();
            }
            inputToOutLayer = signHid;
        }
        int idxMax = 0; //index neuron output dengan nilai sign yang paling besar
        double signMax = -1; //hasil setelah AF lalu dijadikan 0 atau 1
        //menghitung sign untuk setiap neuron hidden sekaligus cari sign terbesar
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
        double[][] insNumeric = normalize(whateverToNumeric(ins)); //ubah ke numeric lalu normalisasi
        int[][] target = buildTarget(insNumeric);
        if (nHidden == 0) {
            singleLayer(insNumeric, target);
        } else if(nHidden > 0) {
            dualLayer(insNumeric, target);
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