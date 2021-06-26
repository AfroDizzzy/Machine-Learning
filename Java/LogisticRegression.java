package SparkExample;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.api.java.JavaRDD;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class LogReg {
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

        // w12-13 s23
        // create train and test splits
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3},seed);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];


        // w12-13 s23
        // define logreg instance
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)             //max iterations
                .setRegParam(0.3)           //lambda
                .setElasticNetParam(0.8);   //alpha

        //fit model
        LogisticRegressionModel lrModel = lr.fit(training);
        System.out.println("Coefficients: " + lrModel.coefficients() + "\nIntercept: " + lrModel.intercept());


        // w12-13 s24
        // extract summary from model
        BinaryLogisticRegressionTrainingSummary trainingSummary = lrModel.binarySummary();
        // get loss per iteration
//        double[] objectiveHistory = trainingSummary.objectiveHistory();
//        for (double lossPerIteration: objectiveHistory) {
//            System.out.println(lossPerIteration);
//        }
        // obtain roc as df and areaUnderROC
//        Dataset<Row> roc = trainingSummary.roc();
//        roc.show();
//        roc.select("FPR").show();
//        System.out.println(trainingSummary.areaUnderROC());
        double train_accuracy = trainingSummary.accuracy();

        /* We need to output the training accuracy */
        System.out.println("Train Error = " + (1.0 - train_accuracy));

        // get threshold of the max f-measure
        Dataset<Row> fMeasure = trainingSummary.fMeasureByThreshold();
        double maxFMeasure = fMeasure.select(functions.max("F-Measure")).head().getDouble(0);
        double bestThreshold = fMeasure.where(fMeasure.col("F-Measure").equalTo(maxFMeasure))
                .select("threshold")
                .head()
                .getDouble(0);
        // set it as threshold for model
        lrModel.setThreshold(bestThreshold);


        // make predictions
        Dataset<Row> predictions = lrModel.transform(test);

        // select example rows to display
//        predictions.show(5);

        // select (prediction, true label) to compute test error
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        // print accuracy
        double test_accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - test_accuracy));

        final long runTime = (System.nanoTime() - startTime)/1000000;
        System.out.println("runtime(ms): " + runTime);

    }
}
