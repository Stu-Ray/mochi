import java.io.*;
import java.util.*;
import java.time.LocalDate;  
import java.time.Period;  
import java.time.temporal.ChronoUnit; 
import java.time.format.DateTimeFormatter;

public class Loadcsv {
	public  int numol;
	public  int numo;
	public  int numc;
	public  int numi;
	public  int numd;
	public  HashMap<Integer,String> dmap2;
	private  List<Integer> o_d_id;
	private  ArrayList<String> o_id;
	private  ArrayList<String> o_c_id;
	private  ArrayList<String> o_item_id;
	private  ArrayList<String> amount;
	private  ArrayList<String> quantity;
	private  ArrayList<String> oldata; 
	private  ArrayList<String> city; 
	private  ArrayList<String> cname; 
	private  ArrayList<String> cstate; 
	private  ArrayList<String> cdiscount;
	private  ArrayList<String> iname;
	private  ArrayList<String> idata; 
	private  ArrayList<String> date;
	public  oorderdata oorder=null; 
	public  orderlinedata orderline=null; 
	public  customerdata customer=null; 
	public  itemdata     item=null;
	public  districtdata district=null;
	public  warehousedata warehouse=null;

	public ArrayList<String> getIname() {
		return iname;
	}
	
	public class oorderdata{
		public List<Integer> oo_id = new ArrayList<>();
		public List<Integer> oo_d_id = new ArrayList<>();
		public List<Integer> oo_c_id = new ArrayList<>();
		public List<Integer> oo_ol_cnt = new ArrayList<>();
		public List<LocalDate> oo_date = new ArrayList<>();
		public List<Long> daysDifference = new ArrayList<>();
		// public int id;
	}
	public  class orderlinedata{
		public List<Integer> ol_d_id = new ArrayList<>();
		public List<Integer> ol_o_id = new ArrayList<>();
		public List<Integer> ol_number = new ArrayList<>();
		public List<Integer> ol_i_id = new ArrayList<>();
		public List<Integer> ol_quantity = new ArrayList<>();
		public List<Double> ol_amount = new ArrayList<>();
		public List<String>  ol_data = new ArrayList<>();
		
	}
	public  class customerdata{
		public List<Integer> c_d_id = new ArrayList<>();
		public List<Integer> c_id = new ArrayList<>();
	// 	public List<Integer> c_balance = new ArrayList<>();
		public List<String>  c_city = new ArrayList<>();
		public List<String>  c_last = new ArrayList<>();
		public List<String>  c_first = new ArrayList<>();
		public List<String>  c_state = new ArrayList<>();
		public List<Double> c_discount = new ArrayList<>();
		
	}
	public class itemdata{
		public List<Integer> i_id = new ArrayList<>();
		public List<String>  i_name = new ArrayList<>();
		public List<Double> i_price = new ArrayList<>();
		public List<String>  i_data = new ArrayList<>();
	}
	public class districtdata{
		public List<Double> d_ytd = new ArrayList<>();
		public List<String>  d_name = new ArrayList<>();
		public List<Integer>  d_next_o_id = new ArrayList<>();
		public List<String>  d_city = new ArrayList<>();
		public List<String>  d_state = new ArrayList<>();
	}
	public class warehousedata{
		public List<Double> w_ytd = new ArrayList<>();
		public List<String>  w_name = new ArrayList<>();
	}
	
	public Loadcsv() {
		o_d_id = new ArrayList<Integer>();
		o_id = new ArrayList<>();
		o_c_id =new ArrayList<>();
		o_item_id = new ArrayList<>();
		amount = new ArrayList<>();
		quantity = new ArrayList<>();
		oldata = new ArrayList<>();
		city = new ArrayList<>();
		cname = new ArrayList<>();
		cstate = new ArrayList<>();
		iname = new ArrayList<>();
		idata = new ArrayList<>();
		cdiscount = new ArrayList<>();
		date = new ArrayList<>();
		
		String csvFile = jTPCC.runfilepath;
		numol=0;
		numd=0;
		HashSet<String> dist = new HashSet<>();       
		HashMap<String,Integer> dmap1 = new HashMap<>();
		dmap2 = new HashMap<>();
		List<String> dcitys = new ArrayList<>();
		List<String> dstates = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(csvFile), "GB2312"))) {
			String line;
			line = br.readLine();	// 读取列名

			while ((line = br.readLine()) != null) {
				numol++;
				// 按逗号分隔行数据
				String[] fields = line.split(",");
				if(!dist.contains(fields[11])) 
				{
					numd++;
					dmap1.put(fields[11],numd);
					dmap2.put(numd, fields[11]);
					o_d_id.add(dmap1.get(fields[11]));
					dcitys.add(fields[8]);
					dstates.add(fields[9]);
				}
				else 
				{
					o_d_id.add(dmap1.get(fields[11]));
				}

				dist.add(fields[11]);
				o_id.add(fields[1]);
				o_c_id.add(fields[5]); 	// original customer id
				o_item_id.add(fields[12]);	// original product id
				amount.add(fields[17]);
				quantity.add(fields[18]);
				oldata.add(fields[6]+" "+fields[7]+" "+fields[8]+" "+fields[9]);
				city.add(fields[8]);
				cname.add(fields[6]);
				cstate.add(fields[9]);
				iname.add(fields[15]+" "+fields[16]);
				cdiscount.add(fields[19]);
				idata.add(fields[13]+" "+fields[14]);
				date.add(fields[2]);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		// 取客户id后5位，排序，赋值（相同内容值一样），还原
		List<String> o_c_id1 = new ArrayList<>();
		List<String> o_c_id2 = new ArrayList<>();
		HashMap<String, Integer> cid_map = new HashMap<>();
		// o_c_id1 = o_c_id;
		for(int i=0; i<numol; i++)
		{
			o_c_id1.add(o_c_id.get(i));
			o_c_id2.add(o_c_id.get(i));
		}		
		Collections.sort(o_c_id1);
		numc=0;	// 表示客户id的数量
		cid_map.put(o_c_id1.get(0), ++numc); // map中的键值对是按顺序插入的，key值是客户id后5位
		// 客户id
		for(int i=1; i<numol; i++)
		{
			if(o_c_id1.get(i).equals(o_c_id1.get(i-1))){
				cid_map.put(o_c_id1.get(i), numc);
			}else{
				cid_map.put(o_c_id1.get(i), ++numc);
			}			
		}
		
		// 客户表的内容
		customer = new customerdata();
		HashSet<Integer>[]  cdids = new HashSet[numc];
		for (int i = 0; i < cdids.length; i++)
		{
			cdids[i] = new HashSet<>();
		}
		HashSet<String> cid = new HashSet<>();
		for(int i=0; i<numol; i++)
		{
			if(cid.contains(o_c_id.get(i)))
			{
				if(cdids[cid_map.get(o_c_id2.get(i))-1].contains(o_d_id.get(i)))
				{
					continue;
				}
			}
			cid.add(o_c_id.get(i));
			cdids[cid_map.get(o_c_id2.get(i))-1].add(o_d_id.get(i));
			customer.c_d_id.add(o_d_id.get(i));
			customer.c_id.add(cid_map.get(o_c_id2.get(i)));
			customer.c_last.add(cname.get(i).substring(0, 1));
			customer.c_first.add(cname.get(i).substring(1));
			customer.c_city.add(city.get(i));
			customer.c_state.add(cstate.get(i));
			customer.c_discount.add(Double.parseDouble(cdiscount.get(i)));
		}
		
		// oorder订单表插入的内容
		oorder = new oorderdata();
		numo=1;
		int cnt=1;
		int j=1;
		int[] a=new int[6]; 	// 不同区域的订单id
		for(int i=0;i<6;i++) {
			a[i]=1;
		}
		oorder.oo_d_id.add(o_d_id.get(0));
		// 初始区域id
		oorder.oo_id.add(a[o_d_id.get(0)-1]++);
		oorder.oo_c_id.add(cid_map.get(o_c_id2.get(0)));
		DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy/M/d");
		oorder.oo_date.add(LocalDate.parse(date.get(0),formatter));
		oorder.daysDifference.add(0L);
		LocalDate start = LocalDate.parse(date.get(0),formatter);
		LocalDate end = LocalDate.parse(date.get(numol-1),formatter);
		long sumday = ChronoUnit.DAYS.between(start,end);

		for(int i=1; i<numol; i++) {
			if(o_id.get(i).equals(o_id.get(i-1))) {
				cnt++;
				continue;
			}
			oorder.oo_d_id.add(o_d_id.get(i));	// 区域id
			numo++;
			if(o_d_id.get(i)==1) {  		// 订单id
				oorder.oo_id.add(a[0]++);
			}else if(o_d_id.get(i)==2) {
				oorder.oo_id.add(a[1]++);
			}else if(o_d_id.get(i)==3) {
				oorder.oo_id.add(a[2]++);
			}else if(o_d_id.get(i)==4) {
				oorder.oo_id.add(a[3]++);
			}else if(o_d_id.get(i)==5) {
				oorder.oo_id.add(a[4]++);
			}else {
				oorder.oo_id.add(a[5]++);
			}
			oorder.oo_c_id.add(cid_map.get(o_c_id2.get(i))); 	// 客户id
			oorder.oo_ol_cnt.add(cnt); 			// 订单项目数量
			cnt=1;
			oorder.oo_date.add(LocalDate.parse(date.get(i),formatter));
			long daysDifference = ChronoUnit.DAYS.between(oorder.oo_date.get(j-1),oorder.oo_date.get(j))*jTPCC.runtime*60000/sumday; //差值
			j++;
			//System.out.print(daysDifference+" ");
			oorder.daysDifference.add(daysDifference);
		}
		oorder.oo_ol_cnt.add(cnt);
		
		// 取商品id后8位，排序，赋值（相同内容值一样），还原
		List<String> o_i_id1 = new ArrayList<>();
		List<String> o_i_id2 = new ArrayList<>();
		HashMap<String, Integer> iid_map = new HashMap<>();
		// o_c_id1 = o_c_id;
		for(int i=0;i<numol;i++){
			o_i_id1.add(o_item_id.get(i));
			o_i_id2.add(o_item_id.get(i));
		}		
		Collections.sort(o_i_id1);
		numi=0;
		iid_map.put(o_i_id1.get(0), ++numi);   	// map中的键值对是按顺序插入的，key值是商品id后8位
		for(int i=1; i<numol; i++)		// 商品id
		{
			if(o_i_id1.get(i).equals(o_i_id1.get(i-1))){
				iid_map.put(o_i_id1.get(i), numi);
			}else{
				iid_map.put(o_i_id1.get(i), ++numi);
			}			
		}

		// orderline订单表插入的内容
		orderline = new orderlinedata();
		int olnumber=1;
		for(int i=0;i<6;i++) {
			a[i]=0;
		}
		//orderline.ol_number.add(olnumber);
		for(int i=0;i<numol;i++){
			orderline.ol_d_id.add(o_d_id.get(i));
			// orderline.ol_o_id.add(a[o_d_id.get(i)-1]++);
			if(i>=1 && o_id.get(i).equals(o_id.get(i-1))){
				orderline.ol_o_id.add(a[o_d_id.get(i)-1]);
				orderline.ol_number.add(++olnumber);
			}else{
				olnumber=1;
				orderline.ol_o_id.add(++a[o_d_id.get(i)-1]);
				orderline.ol_number.add(olnumber);
			}
			orderline.ol_i_id.add(iid_map.get(o_i_id2.get(i)));
			orderline.ol_amount.add(Double.parseDouble(amount.get(i)));
			orderline.ol_quantity.add(Integer.parseInt(quantity.get(i)));
			orderline.ol_data.add(oldata.get(i));
		}

		item = new itemdata();
		HashSet<String> itid = new HashSet<>();
		for(int i=0;i<numol;i++) {
			if(itid.contains(o_item_id.get(i))) {
				continue;
			}
			itid.add(o_item_id.get(i));
			item.i_id.add(iid_map.get(o_i_id2.get(i)));
			item.i_name.add(iname.get(i));
			item.i_price.add(Double.parseDouble(amount.get(i))/Integer.parseInt(quantity.get(i)));
			item.i_data.add(idata.get(i));
		}

		district = new districtdata();
		double[] dnum = new double[numd];
		for(int i=0;i<numd;i++) {
			dnum[i]=0;
		}
		for(int i=0;i<numol;i++) {
			dnum[o_d_id.get(i)-1] += Double.parseDouble(amount.get(i));
		}
		double wytd =0;
		for(int i=0;i<numd;i++) 
		{
			//district.d_name.add();
			district.d_ytd.add(dnum[i]);
			wytd = wytd + dnum[i];
			district.d_name.add(dmap2.get(i+1));
			district.d_next_o_id.add(a[i]);
			district.d_city.add(dcitys.get(i));
			district.d_state.add(dstates.get(i));	
		}
		warehouse = new warehousedata();
		warehouse.w_ytd.add(wytd);
	}
}


