package SparkExample;

import org.apache.spark.SparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.api.java.JavaRDD;

import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;

import org.apache.spark.ml.PipelineModel;


public class DecTree {
    public static void main(String[] args) {

        int seed = Integer.parseInt(args[1]);
//        int seed = 123;
        System.out.println("seed: " + seed);

        // https://stackoverflow.com/questions/27781187/how-to-stop-info-messages-displaying-on-spark-console
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        // start time
        final long startTime = System.nanoTime();

        // configure spark session & context
        SparkSession spark = SparkSession.builder().appName("logistic regression").getOrCreate();
        SparkContext sc = spark.sparkContext();

        // w12-13 s17
        sc.setLogLevel("ERROR");
        JavaRDD<String> lines = sc.textFile(args[0] + "/kdd.data", 0).toJavaRDD();
        JavaRDD<LabeledPoint> linesRDD = lines.map(line -> {
            String[] tokens = line.split(",");
            double[] features = new double[tokens.length - 1];
            for (int i = 0; i < features.length; i++) {
                features[i] = Double.parseDouble(tokens[i]);}
            Vector v = new DenseVector(features);
            if (tokens[features.length].equals("anomaly")) {
                return new LabeledPoint(0.0, v);
            } else {
                return new LabeledPoint(1.0, v);
            }
        });

        Dataset<Row> data = spark.createDataFrame(linesRDD, LabeledPoint.class);
//        data.show();

        // w11-12 s36
        // index labels
        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(data);

        // identify categorical features and index
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(4)
                .fit(data);

        // https://stackoverflow.com/questions/41952815/cannot-recognize-the-dataframe-for-java-on-spark-in-the-intellij-platform
        // split data
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3}, seed);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // train dt model
        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("indexedFeatures");

        // convert indexed labels back to original labels
        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labels());

        // chain indexes and tree in a pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{labelIndexer, featureIndexer, dt, labelConverter});

        // train dt model
        PipelineModel model = pipeline.fit(trainingData);

        // make predictions
        Dataset<Row> predictions = model.transform(testData);
//        predictions.select("predictedLabel", "label", "features").show(10);

        // select (prediction, true label) and compute test error
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        // print test accuracy
        double test_accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - test_accuracy));

        // print train accuracy
        Dataset<Row> test_instances = model.transform(trainingData);
        double train_accuracy = evaluator.evaluate(test_instances);
        System.out.println("Training Error = " + (1.0 - train_accuracy));

        // runtime of program in ms
        final long runTime = (System.nanoTime() - startTime)/1000000;
        System.out.println("runtime(ms): " + runTime);


        DecisionTreeClassificationModel treeModel = (DecisionTreeClassificationModel) (model.stages()[2]);
        // System.out.println("Learned classification tree model:\n" + treeModel.toDebugString());
    }
}
