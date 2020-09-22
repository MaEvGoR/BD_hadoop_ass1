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
import org.json.JSONObject;

public class Search extends Configured implements Tool {
	
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
	
	public static class LengthReader extends Mapper<LongWritable, Text, Text, Text>{
		
		public static Map<Integer, Integer> length_map = new HashMap<Integer, Integer>();
		public static int total_length = 0;
		public static int documents = 0;
		
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
			String s = value.toString();
			String [] split = s.split(" ");
			Integer doc_id = Integer.parseInt(split[0]);
			JSONObject obj = new JSONObject(split[1]);
			int length = obj.getInt("length");
			length_map.put(doc_id, length);
			total_length += length;
			documents ++;
		}
	}
	
	public static class RelevanceFunctionMap extends Mapper<LongWritable, Text, IntWritable, DoubleWritable> {

		public final static double b = 0.75;
		public final static int k1 = 2;
		
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
			
			String s = value.toString();
			JSONObject obj = new JSONObject(s.substring(s.indexOf('{')));
			Integer doc_id = obj.getInt("wiki_id");
			Integer word_id = obj.getInt("word_id");
			Double tf = obj.optDouble("tf");
			Double tf_dtf = obj.optDouble("tf/dtf");
			
			if (QueryVectorizor.query_indeces.contains(word_id)) {
				Integer d = LengthReader.length_map.get(doc_id);
				Double avglen = ((double)LengthReader.total_length)/((double)LengthReader.documents);
				Double nom = tf * (k1 + 1);
				Double denom = tf + k1 * (1 - b + b * (d / avglen));
				Double relevance = tf_dtf * (nom / denom);
				context.write(new IntWritable(doc_id), new DoubleWritable(relevance));
			}			
			
		}
	}
	
	public static class RelevanceFunctionReduce extends Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable>{
		public void reduce(IntWritable key, Iterable<DoubleWritable> list, Context context) throws java.io.IOException, InterruptedException {
			Double sum = 0.0;
			for (DoubleWritable value : list) {
				sum += value.get();
			}
			context.write(key, new DoubleWritable(sum));
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
	
	public static class ReduceRanker extends Reducer<DoubleWritable, IntWritable, IntWritable, Text>{
		
		public static Integer top_N;
		public static FileSystem fs;
		public static Path pt;
		
		public void reduce(DoubleWritable key, Iterable<IntWritable> list, Context context) throws java.io.IOException, InterruptedException {
			if (top_N!=0) {
				for (IntWritable value : list) {
					String doc_title = " ";
					BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(pt)));
					while(br.ready()==true) {
						String line = br.readLine();
						String [] split = line.split(" ");
						Integer doc_id = Integer.parseInt(split[0]);
						if(doc_id == value.get()) {
							String s = split[1];
							JSONObject obj = new JSONObject(s.substring(s.indexOf('{')));
							doc_title = obj.getString("title");
							break;
						}
					}
					context.write(value, new Text(doc_title));
					top_N--;
				}
			}
		}
	}
	
	public int run(String [] args) throws Exception{
		
		JobControl jobControl = new JobControl("chain");
		
		Configuration conf = new Configuration();
		Job vectorize = Job.getInstance(conf, "vectorize query");
		QueryVectorizor.query = args[2];
		ReduceRanker.top_N = Integer.parseInt(args[1]);
		String index_path = args[0];
		
        vectorize.setJarByClass(Search.class);
		vectorize.setMapperClass(QueryVectorizor.class);
		vectorize.setNumReduceTasks(0);
	    vectorize.setOutputKeyClass(Text.class);
	    vectorize.setOutputValueClass(Text.class);
	    String vocab_path = index_path.concat("/vocab");
	    FileInputFormat.addInputPath(vectorize, new Path(vocab_path));
	    FileOutputFormat.setOutputPath(vectorize, new Path("temp/"));
		
	    ControlledJob job1 = new ControlledJob(conf);
	    job1.setJob(vectorize);
	    jobControl.addJob(job1);
	    
	    Configuration conf1 = new Configuration();
		Job read = Job.getInstance(conf, "read document lengths");
		
        read.setJarByClass(Search.class);
		read.setMapperClass(LengthReader.class);
		read.setNumReduceTasks(0);
	    read.setOutputKeyClass(Text.class);
	    read.setOutputValueClass(Text.class);
	    String read_path = index_path.concat("/wiki_info");
	    FileInputFormat.addInputPath(read, new Path(read_path));
	    FileOutputFormat.setOutputPath(read, new Path("temp2/"));
		
	    ControlledJob job2 = new ControlledJob(conf1);
	    job2.setJob(read);
	    job2.addDependingJob(job1);
	    jobControl.addJob(job2);
	    
	    Configuration conf3 = new Configuration();
	    
	    Job assign_relevance = Job.getInstance(conf3, "calculate relevance function");
	    assign_relevance.setJarByClass(Search.class);
	    assign_relevance.setMapperClass(RelevanceFunctionMap.class);
	    assign_relevance.setReducerClass(RelevanceFunctionReduce.class);;
	    assign_relevance.setOutputKeyClass(IntWritable.class);
	    assign_relevance.setOutputValueClass(DoubleWritable.class);
	    String vector_path = index_path.concat("/wiki_vector");
	    FileInputFormat.addInputPath(assign_relevance, new Path(vector_path));
	    FileOutputFormat.setOutputPath(assign_relevance, new Path("temp3/"));
	    
	    ControlledJob job3 = new ControlledJob(conf3);
	    job3.setJob(assign_relevance);
	    job3.addDependingJob(job2);
	    jobControl.addJob(job3);
	    
	    Configuration conf4 = new Configuration();
	    
		FileSystem fs = FileSystem.get(conf4);
		String info_path = index_path.concat("/wiki_info");
		Path pt = new Path(info_path);
		ReduceRanker.fs = fs;
		ReduceRanker.pt = pt;
	    
	    Job rank = Job.getInstance(conf4, "rank the docs");
	    rank.setJarByClass(Search.class);
	    rank.setMapperClass(MapRanker.class);
	    rank.setReducerClass(ReduceRanker.class);
	    rank.setOutputKeyClass(DoubleWritable.class);
	    rank.setOutputValueClass(IntWritable.class);
	    FileInputFormat.addInputPath(rank, new Path("temp3/"));
	    FileOutputFormat.setOutputPath(rank, new Path("output/"));
	    rank.setSortComparatorClass(DecreasingComparator.class);
	    
	    ControlledJob job4 = new ControlledJob(conf3);
	    job4.setJob(rank);
	    job4.addDependingJob(job3);
	    jobControl.addJob(job4);
	    
	    Thread jobControlThread = new Thread(jobControl);
	    jobControlThread.start();
	    while(!jobControl.allFinished()) {
	    	
	    }

	    return 0;
	}
	
	public static void main(String[] args) throws Exception {
		if (args.length!=3) {
			System.out.printf("Incorrect number of arguments\nProper usage: "
					+ "hadoop jar /path/to/jar Search <path to index files> <number of relevant documents to output> <query text>\n");
			System.exit(0);
		}
		try {
			Integer.parseInt(args[1]);
		}
		catch(NumberFormatException nfe)
	     {
			System.out.printf("Incorrect number of arguments\nProper usage: "
					+ "hadoop jar /path/to/jar Search <path to index files> <number of relevant documents to output> <query text>\n");
			System.exit(0);
	     }
		int exitCode = ToolRunner.run(new Search(), args);  
		 System.exit(exitCode);
	}
}
