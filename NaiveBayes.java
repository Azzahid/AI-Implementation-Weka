/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NaiveBayes;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 *
 * @author Silva
 */
public class NaiveBayes extends AbstractClassifier{

    private List<List<List<Double>>> value;
    private List<List<List<Integer>>> Model;
    
    
    @Override
    public void buildClassifier(Instances ins) throws Exception {
        Enumeration<Attribute> enu = ins.enumerateAttributes();
        Attribute attr = enu.nextElement();
        
        if(attr.type() != Attribute.NOMINAL){
            System.out.println("Error : the instances attribute is "+attr.type()+", Filter to Nominal");
        }else{
            value =  new ArrayList<>();
            Model = new ArrayList<>();
            for(int i=0; i<ins.numClasses(); i++){
                value.add(new ArrayList<List<Double>>());
                Model.add(new ArrayList<List<Integer>>());
                for(int j=0; j<ins.numAttributes()-1;j++){
                    int l = 0;
                    int k = 0;
                    value.get(i).add(new ArrayList<Double>());
                    Model.get(i).add(new ArrayList<Integer>());
                    while(k < ins.numInstances()){
                        boolean x = false;
                        if(value.get(i).get(j).isEmpty()){
                            this.value.get(i).get(j).add(l, ins.get(k).value(j));
                            this.Model.get(i).get(j).add(l, 1);
                            l++;
                        }else{
                            //System.out.println(value.get(i).get(j).size());
                            for(int m =0;m<value.get(i).get(j).size();m++){
                                if(ins.get(k).value(j)==this.value.get(i).get(j).get(m)){
                                    if(ins.get(k).classValue()==new Double(i)){
                                        Model.get(i).get(j).set(m, Model.get(i).get(j).get(m)+1);
                                    }  
                                    x = true;
                                }
                            }
                            if(!x){
                                this.value.get(i).get(j).add(l, ins.get(k).value(j));
                                this.Model.get(i).get(j).add(l, 1);
                                l++;
                            }
                        }
                        k++;
                    }
                }
            }
        }
        System.out.println("fish");
    }
    
    public static void main(String[] args) throws Exception {
        Instances i = new DataSource("iris.arff").getDataSet();
        i.setClassIndex(i.numAttributes() - 1); //kelas = atribut terakhir
        Discretize filter = new Discretize();
        filter.setInputFormat(i);
        NumericToNominal x = new NumericToNominal();
        x.setInputFormat(i);
        Instances ix = Filter.useFilter(i, filter);
        Instances ib = Filter.useFilter(i, x);
        NaiveBayes mlp = new NaiveBayes();
        /* ATTENTION: ini sementara di komantarin dulu, pake nilai-nilai default aja */
        //System.out.print("Masukkan jumlah neuron pada hidden layer: ");
        //mlp.setHiddenLayers(new Scanner(System.in).nextInt());
        //mlp.setLearningRate(0.1);
        //mlp.setValidationThreshold(0.5);
        mlp.buildClassifier(ix);
        
        for(int z = 0; z<i.size();z++){
            System.out.println(i.get(z).value(0));
        }
    }
}
