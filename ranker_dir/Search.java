import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
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

// Main search engine class
public class Search extends Configured implements Tool {
	
	// Class to create a vectorized representation of a query
	public static class QueryVectorizor extends Mapper<LongWritable, Text, Text, Text>{
		
		// Storing the name of the query
		public static String query = new String();
		
		// Storing word IDs that are encountered in a query
		public static Set<Integer> query_indeces = new HashSet<Integer>();
		
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
				String s = value.toString();
				JSONObject obj = new JSONObject(s.substring(s.indexOf('{')));
				String word = obj.getString("word");
				
				// Processing every word in a query and storing its ID
				String [] query_split = query.split(" ");
				for (String temp : query_split) {
					temp.replaceAll("[^a-zA-Z0-9_-]", "");
					if (temp.compareTo(word) == 0) {
						query_indeces.add(obj.getInt("id"));
					}
				} 
		}		
	}
	
	// Class to read lengths of files into memory to be used later
	public static class LengthReader extends Mapper<LongWritable, Text, Text, Text>{
		
		// Map between word IDs and its length
		public static Map<Integer, Integer> length_map = new HashMap<Integer, Integer>();
		public static int total_length = 0;
		public static int documents = 0;
		
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
				String s = value.toString();
				JSONObject obj = new JSONObject(s.substring(s.indexOf('{')));
				int length = obj.getInt("length");
				Integer doc_id = obj.getInt("wiki_id");
				length_map.put(doc_id, length);
				total_length += length;
				documents ++;
		}
	}
	
	// Mapper for relevance function calculator
	public static class RelevanceFunctionMap extends Mapper<LongWritable, Text, IntWritable, DoubleWritable> {
		
		// Preset parameters for Okapi BM25
		public final static double b = 0.75;
		public final static int k1 = 2;
		
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
				
				// Obtaining info about a current file and word
				String s = value.toString();
				JSONObject obj = new JSONObject(s.substring(s.indexOf('{')));
				Integer doc_id = obj.getInt("wiki_id");
				Integer word_id = obj.getInt("word_id");
				Double tf = obj.optDouble("tf");
				Double tf_dtf = obj.optDouble("tf/idf");
				
				// Making sure that this word is present in a query and recording the relevance
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
	
	// Relevance function reducer that sums up all relevances for a particular document
	public static class RelevanceFunctionReduce extends Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable>{
		public void reduce(IntWritable key, Iterable<DoubleWritable> list, Context context) throws java.io.IOException, InterruptedException {
			Double sum = 0.0;
			for (DoubleWritable value : list) {
				sum += value.get();
			}
			context.write(key, new DoubleWritable(sum));
		}
	}
	
	// Mapper for ranking the documents according to the relevance
	public static class MapRanker extends Mapper<LongWritable, Text, DoubleWritable, IntWritable>{
		public void map(LongWritable key, Text value, Context context) throws java.io.IOException, InterruptedException {
			String line = value.toString();
			String[] tokens = line.split("	"); 
			int keypart = Integer.parseInt(tokens[0]);
			Double valuePart = Double.parseDouble(tokens[1]);
			context.write(new DoubleWritable(valuePart), new IntWritable(keypart));
		}
	}
	
	// The ranking itself is performed in the intermediate step between map and reduce
	// This class makes sure we output only the required number of document titles
	public static class ReduceRanker extends Reducer<DoubleWritable, IntWritable, IntWritable, Text>{
		
		public static Integer top_N = 0;
		public static FileSystem fs;
		public static Path pt;
		
		public void reduce(DoubleWritable key, Iterable<IntWritable> list, Context context) throws java.io.IOException, InterruptedException {
				if (top_N!=0) {
					for (IntWritable value : list) {
						String doc_title = " ";
						RemoteIterator<LocatedFileStatus> fileStatusListIterator = fs.listFiles(
					            pt, true);
					    while(fileStatusListIterator.hasNext()){
					        LocatedFileStatus fileStatus = fileStatusListIterator.next();
					        Path pt2 = fileStatus.getPath();
					        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(pt2)));
					        while(br.ready()==true) {
								String line = br.readLine();
								JSONObject obj = new JSONObject(line.substring(line.indexOf('{')));
								Integer doc_id = obj.getInt("wiki_id");
								if(doc_id == value.get()) {
									doc_title = obj.getString("title");
									break;
								}
							}
					        if (doc_title.compareTo(" ")!=0) {
					        	break;
					        };
					    }
						context.write(value, new Text(doc_title));
						top_N--;
					}
				}
		}
	}
	
	// Controller function for all the jobs
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
	    String vocab_path = index_path.concat("/indexes/vocab");
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
	    String vector_path = index_path.concat("/indexes/wiki_vector");
	    FileInputFormat.addInputPath(assign_relevance, new Path(vector_path));
	    FileOutputFormat.setOutputPath(assign_relevance, new Path("temp3/"));
	    
	    ControlledJob job3 = new ControlledJob(conf3);
	    job3.setJob(assign_relevance);
	    job3.addDependingJob(job2);
	    jobControl.addJob(job3);
	    
	    Configuration conf4 = new Configuration();
	    
		FileSystem fs = FileSystem.get(conf4);
		String info_path = index_path.concat("/wiki_info/");
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
	    
	    ControlledJob job4 = new ControlledJob(conf4);
	    job4.setJob(rank);
	    job4.addDependingJob(job3);
	    jobControl.addJob(job4);
	    
	    Thread jobControlThread = new Thread(jobControl);
	    jobControlThread.start();
	    while(!jobControl.allFinished()) {
	    	
	    }
	    
	    Configuration conf5 = new Configuration();
	    fs = FileSystem.get(conf5);
	    RemoteIterator<LocatedFileStatus> fileStatusListIterator = fs.listFiles(
	            new Path("output"), true);
	    while(fileStatusListIterator.hasNext()){
	        LocatedFileStatus fileStatus = fileStatusListIterator.next();
	        Path pt2 = fileStatus.getPath();
	        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(pt2)));
	        while(br.ready()) {
	        	String line = br.readLine();
				String [] split = line.split("	");
				System.out.println(split[1]);
	        }
	    }
	    
	    
	    return 0;
	}
	
	// Main & handling of incorrect argument usage
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
