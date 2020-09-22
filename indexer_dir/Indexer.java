import java.io.IOException;
import java.util.StringTokenizer;
import java.util.Arrays;
import java.util.Iterator;

import org.json.JSONObject;
import org.json.JSONException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;

import org.apache.hadoop.mapred.jobcontrol.JobControl;
import org.apache.hadoop.mapreduce.lib.jobcontrol.ControlledJob;


public class Indexer {

    public static class WikiInfo extends Mapper<Object, Text, IntWritable, Text>{

//        private final static IntWritable doc_id = new IntWritable(1);
//        private Text url = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            try {
                JSONObject one_doc = new JSONObject(value.toString().replace('"', '\"'));

                JSONObject wiki_id_info = new JSONObject();
                wiki_id_info.put("wiki_id", one_doc.getString("id"));
                wiki_id_info.put("url", one_doc.getString("url"));
                wiki_id_info.put("title", one_doc.getString("title"));

                IntWritable doc_id = new IntWritable(one_doc.getInt("id"));
                Text info = new Text(wiki_id_info.toString());
                context.write(doc_id, info);
            }
            catch (JSONException e) {
                System.out.println("Not working");
            }
        }

    }

    public static class WordVocab extends Mapper<Object, Text, Text, Text>{

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            try {
                JSONObject one_doc = new JSONObject(value.toString().replace('"', '\"'));

                String my_text = one_doc.getString("text").replaceAll("[^a-z]", " ");

                StringTokenizer itr = new StringTokenizer(my_text);
                Text doc_id = new Text(one_doc.getString("id"));
//                IntWritable doc_id = new IntWritable(one_doc.getInt("id"));

                while (itr.hasMoreTokens()) {
                    String current_word = itr.nextToken().toLowerCase();
                    if (current_word != "") {
                        Text word = new Text();
                        word.set(current_word);
                        context.write(word, doc_id);
                    }
                }
            }
            catch (JSONException e) {
                context.write(new Text("not working"), new Text("1"));
            }
        }
    }

    public static class CombinedReducer extends Reducer<Text, Text, Text, Text> {

        MultipleOutputs multipleOutputs;

        protected void setup(Context context) throws IOException, InterruptedException {
            multipleOutputs = new MultipleOutputs(context);
        }

        private int word_id = 0;

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int current_word_id = word_id;
            word_id = word_id + 1;

            // init idf
            int idf = 0;

            // init json for word for easy storing for me
            JSONObject word_info = new JSONObject();
            // get unique wiki_ids
            for (Text wiki_id : values) {
                int current_wiki_id = Integer.parseInt(wiki_id.toString());
//                int current_wiki_id = wiki_id.get();
                try {
                    if (word_info.has(Integer.toString(current_wiki_id))) {
                        JSONObject wiki_word_info = word_info.getJSONObject(Integer.toString(current_wiki_id));
                        wiki_word_info.put("tf", wiki_word_info.getInt("tf") + 1);
                        word_info.put(Integer.toString(current_wiki_id), wiki_word_info);
                    } else {
                        idf = idf + 1;
                        JSONObject new_wiki_word_info = new JSONObject();
                        new_wiki_word_info.put("tf", 1);
                        word_info.put(Integer.toString(current_wiki_id), new_wiki_word_info);
                    }
                } catch (JSONException e) {
                    System.out.println("not working1");
                }
            }

            try {
                Iterator<String> word_info_keys = word_info.keys();

                while (word_info_keys.hasNext()) {
                    String current_key = word_info_keys.next();
                    JSONObject wiki_word_info = word_info.getJSONObject(current_key);
                    wiki_word_info.put("tf/idf", wiki_word_info.getInt("tf") / (float) idf);
                    wiki_word_info.put("wiki_id", current_key);
                    wiki_word_info.put("word_id", current_word_id);
                    word_info.put(current_key, wiki_word_info);

                    multipleOutputs.write("wikivector", new Text(current_key), new Text(wiki_word_info.toString()), "wiki_vector/");
                }
            } catch (JSONException e) {
                System.out.println("not working2");
            }

            try {
                JSONObject json_vocab = new JSONObject();
                json_vocab.put("id", current_word_id);
                json_vocab.put("word", key);
                json_vocab.put("idf", idf);

//                context.write(key, new Text(json_vocab.toString()));
                multipleOutputs.write("vocab", key, new Text(json_vocab.toString()), "vocab/");
            } catch (JSONException e) {
                context.write(key, new Text("not working3"));
            }
        }

    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("error");
            System.exit(-1);
        }
        // init and configure jobcontrol for multiple tasks
        JobControl jobControl = new JobControl("chain");

        // first job creation
        Configuration conf = new Configuration();

        Job WikiIndexJob = Job.getInstance(conf, "indexer");

        WikiIndexJob.setJarByClass(Indexer.class);
        WikiIndexJob.setMapperClass(WikiInfo.class);

        WikiIndexJob.setNumReduceTasks(0);

        WikiIndexJob.setOutputKeyClass(IntWritable.class);
        WikiIndexJob.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(WikiIndexJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(WikiIndexJob, new Path(args[1]+"/wiki_info"));

        ControlledJob job1 = new ControlledJob(conf);
        job1.setJob(WikiIndexJob);
        jobControl.addJob(job1);


        // second job creation

        Configuration conf2 = new Configuration();

        Job VocabMakerJob = Job.getInstance(conf2, "vocab_maker");

        VocabMakerJob.setJarByClass(Indexer.class);
        VocabMakerJob.setMapperClass(WordVocab.class);

        VocabMakerJob.setNumReduceTasks(1);

        VocabMakerJob.setReducerClass(CombinedReducer.class);

//        VocabMakerJob.setMapOutputKeyClass(Text.class);
//        VocabMakerJob.setMapOutputValueClass(IntWritable.class);

        VocabMakerJob.setOutputKeyClass(Text.class);
        VocabMakerJob.setOutputValueClass(Text.class);

        // config for multiple outputs

        MultipleOutputs.addNamedOutput(VocabMakerJob, "vocab", TextOutputFormat.class, Text.class, Text.class);
        MultipleOutputs.addNamedOutput(VocabMakerJob, "wikivector", TextOutputFormat.class, Text.class, Text.class);

        FileInputFormat.addInputPath(VocabMakerJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(VocabMakerJob, new Path(args[1] + "/indexes"));



        ControlledJob job2 = new ControlledJob(conf2);
        job2.setJob(VocabMakerJob);
//        job2.addDependingJob(job1);
        jobControl.addJob(job2);

        Thread jobControlThread = new Thread(jobControl);
        jobControlThread.start();
        while(!jobControl.allFinished()) { }
        System.exit(0);
    }
}