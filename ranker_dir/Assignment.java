import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.LongWritable.DecreasingComparator;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.jobcontrol.JobControl;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.jobcontrol.ControlledJob;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class Assignment extends Configured implements Tool {
	
	public static class QueryVectorizor extends Mapper<LongWritable, Text, Text, Text>{
		
		public static String query;
		public static Set<Integer> query_indeces = new HashSet<Integer>();
		
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			StringTokenizer itr = new StringTokenizer(value.toString());
			String word_id = itr.nextToken();
			String word = itr.nextToken();
			String [] query_split = query.split(" ");
			for (String temp : query_split) {
				if (temp.compareTo(word) == 0) {
					query_indeces.add(Integer.parseInt(word_id));
				}
			}
		}		
	}
	
	public static class RelevanceFunction extends Mapper<LongWritable, Text, Text, DoubleWritable> {
		
		public final static int avglength = 9;
		public final static double b = 0.75;
		public final static int k1 = 2;
		
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
			
			StringTokenizer itr = new StringTokenizer(value.toString());
			String doc_id = itr.nextToken();
			String doc_len = itr.nextToken();
			
			Map<Integer, Double[]> map = new HashMap<Integer, Double[]>();
			while(itr.hasMoreTokens()) {
				String word_id = itr.nextToken();
				String tf = itr.nextToken();
				String tf_idf = itr.nextToken();
				Double[] array = new Double[2];
				array[0] = Double.parseDouble(tf);
				array[1] = Double.parseDouble(tf_idf);
				map.put(Integer.parseInt(word_id), array);
			}
			Set<Integer> keys = map.keySet();
			keys.retainAll(QueryVectorizor.query_indeces);
			
			Double relevance = 0.0;
			
			for (Integer word_key : keys) {
				Double d = map.get(word_key)[0];
				Double idf = map.get(word_key)[1];
				Double nom = d*(k1+1);
				Double denom = d+k1*(1-b+b*(Integer.parseInt(doc_len)/avglength));
				relevance += idf*(nom/denom);
			}
			
			context.write(new Text(doc_id), new DoubleWritable(relevance));
		}
	}
	
	public static class MapRanker extends Mapper<LongWritable, Text, DoubleWritable, IntWritable>{
		public void map(LongWritable key, Text value, Context context) throws java.io.IOException, InterruptedException {
			String line = value.toString();
			String[] tokens = line.split("	"); 
			int keypart = Integer.parseInt(tokens[0]);
			Double valuePart = Double.parseDouble(tokens[1]);
			context.write(new DoubleWritable(valuePart), new IntWritable(keypart));
		}
	}
	
	public static class ReduceRanker extends Reducer<DoubleWritable, IntWritable, IntWritable, DoubleWritable>{
		public void reduce(DoubleWritable key, Iterable<IntWritable> list, Context context) throws java.io.IOException, InterruptedException {
			for (IntWritable value : list) {
				System.out.printf("doc %d: %f\n", value.get(), key.get());
				context.write(value, key);
			}
		}
	}
	
	public int run(String [] args) throws Exception{
		
		JobControl jobControl = new JobControl("chain");
		
		Configuration conf = new Configuration();
		Job vectorize = Job.getInstance(conf, "vectorize query");
		Path pt = new Path("indexer/query.txt");
		FileSystem fs = FileSystem.get(conf);
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(pt)));
        String line;
        line = br.readLine();
        QueryVectorizor.query = line;
		
        vectorize.setJarByClass(Assignment.class);
		vectorize.setMapperClass(QueryVectorizor.class);
		vectorize.setNumReduceTasks(0);
	    vectorize.setOutputKeyClass(Text.class);
	    vectorize.setOutputValueClass(Text.class);
	    FileInputFormat.addInputPath(vectorize, new Path("indexer/vocabulary.txt"));
	    FileOutputFormat.setOutputPath(vectorize, new Path("temp/"));
		
	    ControlledJob job1 = new ControlledJob(conf);
	    job1.setJob(vectorize);
	    jobControl.addJob(job1);
	    
		Configuration conf2 = new Configuration();
	    
	    Job assign_relevance = Job.getInstance(conf2, "calculate relevance function");
	    assign_relevance.setJarByClass(Assignment.class);
	    assign_relevance.setMapperClass(RelevanceFunction.class);
	    assign_relevance.setNumReduceTasks(0);
	    assign_relevance.setOutputKeyClass(Text.class);
	    assign_relevance.setOutputValueClass(DoubleWritable.class);
	    FileInputFormat.addInputPath(assign_relevance, new Path("indexer/tf_idf.txt"));
	    FileOutputFormat.setOutputPath(assign_relevance, new Path("temp2/"));
	    
	    ControlledJob job2 = new ControlledJob(conf2);
	    job2.setJob(assign_relevance);
	    job2.addDependingJob(job1);
	    jobControl.addJob(job2);
	    
	    Configuration conf3 = new Configuration();
	    Job rank = Job.getInstance(conf3, "rank the docs");
	    rank.setJarByClass(Assignment.class);
	    rank.setMapperClass(MapRanker.class);
	    rank.setReducerClass(ReduceRanker.class);
	    rank.setOutputKeyClass(DoubleWritable.class);
	    rank.setOutputValueClass(IntWritable.class);
	    FileInputFormat.addInputPath(rank, new Path("temp2/"));
	    FileOutputFormat.setOutputPath(rank, new Path("output/"));
	    rank.setSortComparatorClass(DecreasingComparator.class);
	    
	    ControlledJob job3 = new ControlledJob(conf3);
	    job3.setJob(rank);
	    job3.addDependingJob(job2);
	    jobControl.addJob(job3);
	    
	    Thread jobControlThread = new Thread(jobControl);
	    jobControlThread.start();
	    while(!jobControl.allFinished()) {
	    	
	    }
	    return 0;
	}
	
	public static void main(String[] args) throws Exception {
		int exitCode = ToolRunner.run(new Assignment(), args);  
		 System.exit(exitCode);
	}
}
