/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NaiveBayes;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.lang.Integer;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
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

    private List<List<Double>> value;
    private List<List<List<Double>>> Model;
    private List<Double> Class;
    private Double totalInstances;
    
    @Override
    public void buildClassifier(Instances ins) throws Exception {
        Enumeration<Attribute> enu = ins.enumerateAttributes();
        Attribute attr = enu.nextElement();
        getClassTotalInstances(ins);
        totalInstances = new Double(ins.numInstances());
        if(attr.type() != Attribute.NOMINAL){
            System.out.println("Error : the instances attribute is "+attr.type()+", Filter to Nominal");
        }else{
            value =  new ArrayList<>();
            Model = new ArrayList<>();
            
            for(int j=0; j<ins.numAttributes()-1;j++){
                value.add(new ArrayList<>());
                Model.add(new ArrayList<List<Double>>());
                int l = 0;
                for(int k=0;k<ins.numInstances();k++ ){
                    int classval=(int) ins.get(k).classValue();
                    boolean x = false;
                    if(value.get(j).isEmpty()){
                        this.value.get(j).add(l, ins.get(k).value(j));
                        this.Model.get(j).add(l, new ArrayList<Double>());
                        for(int i =0; i<ins.numClasses();i++){
                            this.Model.get(j).get(l).add(i,new Double(0));
                        }
                        l++;
                    }
                    //System.out.println(value.get(i).get(j).size());
                    for(int m =0;m<value.get(j).size();m++){
                        if(ins.get(k).value(j)==this.value.get(j).get(m)){
                            Model.get(j).get(m).set(classval, Model.get(j).get(m).get(classval)+1);
                            x = true;
                        }
                    }
                    if(!x){
                        this.value.get(j).add(l, ins.get(k).value(j));
                        this.Model.get(j).add(l, new ArrayList<Double>());
                        for(int i =0; i<ins.numClasses();i++){
                            this.Model.get(j).get(l).add(i,new Double(0));
                        }
                        l++;
                        for(int m =0;m<value.get(j).size();m++){
                        if(ins.get(k).value(j)==this.value.get(j).get(m)){
                            Model.get(j).get(m).set(classval, Model.get(j).get(m).get(classval)+1);
                            x = true;
                        }
                    }
                    }
                }
            }
        }
        
        //System.out.println("fish");
    }
    
    private void getClassTotalInstances(Instances ins){
        Class = new ArrayList<>();
        for(int i=0;i<ins.numClasses();i++){
            Class.add(i,0.0);
        }
        for(int i=0;i<ins.numInstances();i++){
            int classval = (int)ins.get(i).classValue();
            Class.set(classval, Class.get(classval)+1.0);
        }
    }
    
    @Override
    public double classifyInstance(Instance ins) throws Exception {
        double result = 0.0;
        Double classins[] = new Double[ins.numClasses()];
        for(int i =0; i<ins.numClasses();i++){
            classins[i] = new Double((double)this.Class.get(i)/(double)this.totalInstances);
            for(int j=0;j<ins.numAttributes()-1;j++){
                classins[i] *= (Model.get(j).get(getIndex(ins.value(j), j)).get(i)/Class.get(i));
            }
        }
        double max = classins[0];
        for(int i =0;i<classins.length;i++){
            if(classins[i]>max){
                max = classins[i];
                result = new Double(i);
            }
        }
        return result;
    }
    
    private int getIndex(double ins, int x){
        for(int i = 0; i<value.get(x).size();i++){
            if(value.get(x).get(i) == ins){
                return i;
            }
        }
        return 0;
    }
    
}
