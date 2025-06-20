
/*
 * LoadDataWorker - Class to load one Warehouse (or in a special case
 * the ITEM table).
 *
 * Copyright (C) 2016, Denis Lussier
 * Copyright (C) 2016, Jan Wieck
 *
 */
import java.sql.*;
import java.util.*;
import java.io.*;

public class LoadDataWorker implements Runnable {
	private int worker;
	private Connection dbConn;
	private jTPCCRandom rnd;

	private StringBuffer sb;
	private Formatter fmt;

	private boolean writeCSV = false;
	private String csvNull = null;

	private PreparedStatement stmtConfig = null;
	private PreparedStatement stmtItem = null;
	private PreparedStatement stmtWarehouse = null;
	private PreparedStatement stmtDistrict = null;
	private PreparedStatement stmtStock = null;
	private PreparedStatement stmtCustomer = null;
	private PreparedStatement stmtHistory = null;
	private PreparedStatement stmtOrder = null;
	private PreparedStatement stmtOrderLine = null;
	private PreparedStatement stmtNewOrder = null;

	private StringBuffer sbConfig = null;
	private Formatter fmtConfig = null;
	private StringBuffer sbItem = null;
	private Formatter fmtItem = null;
	private StringBuffer sbWarehouse = null;
	private Formatter fmtWarehouse = null;
	private StringBuffer sbDistrict = null;
	private Formatter fmtDistrict = null;
	private StringBuffer sbStock = null;
	private Formatter fmtStock = null;
	private StringBuffer sbCustomer = null;
	private Formatter fmtCustomer = null;
	private StringBuffer sbHistory = null;
	private Formatter fmtHistory = null;
	private StringBuffer sbOrder = null;
	private Formatter fmtOrder = null;
	private StringBuffer sbOrderLine = null;
	private Formatter fmtOrderLine = null;
	private StringBuffer sbNewOrder = null;
	private Formatter fmtNewOrder = null;

	private int numol;
	private int numo;
	private int numc;
	private int numi;
	private int numd;

	private List<Integer> o_d_id;
	private ArrayList<String> o_id;
	private ArrayList<String> o_c_id;
	private ArrayList<String> o_item_id;
	private ArrayList<String> amount;
	private ArrayList<String> quantity;
	private ArrayList<String> oldata;
	private ArrayList<String> city;
	private ArrayList<String> cname;
	private ArrayList<String> cstate;
	private ArrayList<String> cdiscount;
	private ArrayList<String> iname;
	private ArrayList<String> idata;

	// private static ArrayList<String> amount;
	// private static ArrayList<String> quantity;
	private oorderdata oorder = null;
	private orderlinedata orderline = null;
	private customerdata customer = null;
	private itemdata item = null;
	private districtdata district = null;
	private warehousedata warehouse = null;

	private class oorderdata {
		public List<Integer> oo_id = new ArrayList<>();
		public List<Integer> oo_d_id = new ArrayList<>();
		public List<Integer> oo_c_id = new ArrayList<>();
		public List<Integer> oo_ol_cnt = new ArrayList<>();
		// public int id;
	}

	private class orderlinedata {
		public List<Integer> ol_d_id = new ArrayList<>();
		public List<Integer> ol_o_id = new ArrayList<>();
		public List<Integer> ol_number = new ArrayList<>();
		public List<Integer> ol_i_id = new ArrayList<>();
		public List<Integer> ol_quantity = new ArrayList<>();
		public List<Double> ol_amount = new ArrayList<>();
		public List<String> ol_data = new ArrayList<>();

	}

	private class customerdata {
		public List<Integer> c_d_id = new ArrayList<>();
		public List<Integer> c_id = new ArrayList<>();
		// public List<Integer> c_balance = new ArrayList<>();
		public List<String> c_city = new ArrayList<>();
		public List<String> c_last = new ArrayList<>();
		public List<String> c_first = new ArrayList<>();
		public List<String> c_state = new ArrayList<>();
		public List<Double> c_discount = new ArrayList<>();

	}

	private class itemdata {
		public List<Integer> i_id = new ArrayList<>();
		public List<String> i_name = new ArrayList<>();
		public List<Double> i_price = new ArrayList<>();
		public List<String> i_data = new ArrayList<>();
		// public List<Integer> i_im_id = new ArrayList<>();
	}

	private class districtdata {
		public List<Double> d_ytd = new ArrayList<>();
		public List<String> d_name = new ArrayList<>();
		public List<Integer> d_next_o_id = new ArrayList<>();
		public List<String> d_city = new ArrayList<>();
		public List<String> d_state = new ArrayList<>();
	}

	private class warehousedata {
		public List<Double> w_ytd = new ArrayList<>();
		public List<String> w_name = new ArrayList<>();
	}

	LoadDataWorker(int worker, String csvNull, jTPCCRandom rnd) {
		this.worker = worker;
		this.csvNull = csvNull;
		this.rnd = rnd;

		this.sb = new StringBuffer();
		this.fmt = new Formatter(sb);
		this.writeCSV = true;

		this.sbConfig = new StringBuffer();
		this.fmtConfig = new Formatter(sbConfig);
		this.sbItem = new StringBuffer();
		this.fmtItem = new Formatter(sbItem);
		this.sbWarehouse = new StringBuffer();
		this.fmtWarehouse = new Formatter(sbWarehouse);
		this.sbDistrict = new StringBuffer();
		this.fmtDistrict = new Formatter(sbDistrict);
		this.sbStock = new StringBuffer();
		this.fmtStock = new Formatter(sbStock);
		this.sbCustomer = new StringBuffer();
		this.fmtCustomer = new Formatter(sbCustomer);
		this.sbHistory = new StringBuffer();
		this.fmtHistory = new Formatter(sbHistory);
		this.sbOrder = new StringBuffer();
		this.fmtOrder = new Formatter(sbOrder);
		this.sbOrderLine = new StringBuffer();
		this.fmtOrderLine = new Formatter(sbOrderLine);
		this.sbNewOrder = new StringBuffer();
		this.fmtNewOrder = new Formatter(sbNewOrder);
	}

	LoadDataWorker(int worker, Connection dbConn, jTPCCRandom rnd) throws SQLException {
		this.worker = worker;
		this.dbConn = dbConn;
		this.rnd = rnd;

		this.sb = new StringBuffer();
		this.fmt = new Formatter(sb);

		stmtConfig = dbConn.prepareStatement(
				"INSERT INTO bmsql_config (" +
						"  cfg_name, cfg_value) " +
						"VALUES (?, ?)");
		stmtItem = dbConn.prepareStatement(
				"INSERT INTO bmsql_item (" +
						"  i_id, i_im_id, i_name, i_price, i_data) " +
						"VALUES (?, ?, ?, ?, ?)");
		stmtWarehouse = dbConn.prepareStatement(
				"INSERT INTO bmsql_warehouse (" +
						"  w_id, w_name, w_street_1, w_street_2, w_city, " +
						"  w_state, w_zip, w_tax, w_ytd) " +
						"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
		stmtStock = dbConn.prepareStatement(
				"INSERT INTO bmsql_stock (" +
						"  s_i_id, s_w_id, s_quantity, s_dist_01, s_dist_02, " +
						"  s_dist_03, s_dist_04, s_dist_05, s_dist_06, " +
						"  s_dist_07, s_dist_08, s_dist_09, s_dist_10, " +
						"  s_ytd, s_order_cnt, s_remote_cnt, s_data) " +
						"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
		stmtDistrict = dbConn.prepareStatement(
				"INSERT INTO bmsql_district (" +
						"  d_id, d_w_id, d_name, d_street_1, d_street_2, " +
						"  d_city, d_state, d_zip, d_tax, d_ytd, d_next_o_id) " +
						"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
		stmtCustomer = dbConn.prepareStatement(
				"INSERT INTO bmsql_customer (" +
						"  c_id, c_d_id, c_w_id, c_first, c_middle, c_last, " +
						"  c_street_1, c_street_2, c_city, c_state, c_zip, " +
						"  c_phone, c_since, c_credit, c_credit_lim, c_discount, " +
						"  c_balance, c_ytd_payment, c_payment_cnt, " +
						"  c_delivery_cnt, c_data) " +
						"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, " +
						"        ?, ?, ?, ?, ?, ?)");
		stmtHistory = dbConn.prepareStatement(
				"INSERT INTO bmsql_history (" +
						"  hist_id, h_c_id, h_c_d_id, h_c_w_id, h_d_id, h_w_id, " +
						"  h_date, h_amount, h_data) " +
						"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
		stmtOrder = dbConn.prepareStatement(
				"INSERT INTO bmsql_oorder (" +
						"  o_id, o_d_id, o_w_id, o_c_id, o_entry_d, " +
						"  o_carrier_id, o_ol_cnt, o_all_local) " +
						"VALUES (?, ?, ?, ?, ?, ?, ?, ?)");
		stmtOrderLine = dbConn.prepareStatement(
				"INSERT INTO bmsql_order_line (" +
						"  ol_o_id, ol_d_id, ol_w_id, ol_number, ol_i_id, " +
						"  ol_supply_w_id, ol_delivery_d, ol_quantity, " +
						"  ol_amount, ol_dist_info) " +
						"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
		stmtNewOrder = dbConn.prepareStatement(
				"INSERT INTO bmsql_new_order (" +
						"  no_o_id, no_d_id, no_w_id) " +
						"VALUES (?, ?, ?)");

		o_d_id = new ArrayList<Integer>();
		o_id = new ArrayList<>();
		o_c_id = new ArrayList<>();
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

		String csvFile = LoadData.filepath; // LoadData.filepath; //"/home/lyx/桌面/data2-GB.csv"; // CSV 文件路径
		numol = 0;
		numd = 0;
		HashSet<String> dist = new HashSet<>();
		HashMap<String, Integer> dmap1 = new HashMap<>();
		HashMap<Integer, String> dmap2 = new HashMap<>();
		List<String> dcitys = new ArrayList<>();
		List<String> dstates = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(csvFile), "GB2312"))) {
			String line;
			line = br.readLine(); // 读取列名
			while ((line = br.readLine()) != null) {
				numol++;
				// 按逗号分隔行数据
				String[] fields = line.split(",");
				// System.out.println(fields[11]);
				if (!dist.contains(fields[11])) {
					numd++;
					dmap1.put(fields[11], numd);
					dmap2.put(numd, fields[11]);
					o_d_id.add(dmap1.get(fields[11]));
					dcitys.add(fields[8]);
					dstates.add(fields[9]);
					// dname.add(fields[11]);
				} else {
					o_d_id.add(dmap1.get(fields[11]));
				}

				/*
				 * if(fields[11].equals("华东")) {
				 * o_d_id.add(1);
				 * }else if(fields[11].equals("西南")) {
				 * o_d_id.add(2);
				 * }else if(fields[11].equals("西北")) {
				 * o_d_id.add(3);
				 * }else if(fields[11].equals("中南")) {
				 * o_d_id.add(4);
				 * }else if(fields[11].equals("华北")) {
				 * o_d_id.add(5);
				 * }else {
				 * o_d_id.add(6);
				 * }
				 */
				dist.add(fields[11]);
				o_id.add(fields[1]);
				o_c_id.add(fields[5]);
				o_item_id.add(fields[12]);
				amount.add(fields[17]);
				quantity.add(fields[18]);
				oldata.add(fields[6] + " " + fields[7] + " " + fields[8] + " " + fields[9]);
				city.add(fields[8]);
				cname.add(fields[6]);
				cstate.add(fields[9]);
				iname.add(fields[15] + " " + fields[16]);
				cdiscount.add(fields[19]);
				idata.add(fields[13] + " " + fields[14]);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		// 取客户id后5位，排序，赋值（相同内容值一样），还原
		List<String> o_c_id1 = new ArrayList<>();
		List<String> o_c_id2 = new ArrayList<>();
		HashMap<String, Integer> cid_map = new HashMap<>();
		// o_c_id1 = o_c_id;
		for (int i = 0; i < numol; i++) {
			o_c_id1.add(o_c_id.get(i));
			o_c_id2.add(o_c_id.get(i)); // .substring(o_c_id.get(i).length()-5)
		}
		Collections.sort(o_c_id1);
		numc = 0; // 表示客户id的数量
		cid_map.put(o_c_id1.get(0), ++numc); // map中的键值对是按顺序插入的，key值是客户id后5位
		for (int i = 1; i < numol; i++) { // 客户id
			if (o_c_id1.get(i).equals(o_c_id1.get(i - 1))) {
				cid_map.put(o_c_id1.get(i), numc);
			} else {
				cid_map.put(o_c_id1.get(i), ++numc);
			}
		}

		// 客户表的内容
		customer = new customerdata();
		// HashSet<String>[] citys = new HashSet[numc];
		HashSet<Integer>[] cdids = new HashSet[numc];
		// for (int i = 0; i < citys.length; i++) {
		// citys[i] = new HashSet<>();
		// }
		for (int i = 0; i < cdids.length; i++) {
			cdids[i] = new HashSet<>();
		}
		numc = 0;
		HashSet<String> cid = new HashSet<>();
		for (int i = 0; i < numol; i++) {
			if (cid.contains(o_c_id.get(i))) {
				if (cdids[cid_map.get(o_c_id2.get(i)) - 1].contains(o_d_id.get(i))) {
					continue;
				}
				// if(citys[cid_map.get(o_c_id2.get(i))-1].contains(city.get(i))) {
				// continue;
				// }
			}
			numc++;
			cid.add(o_c_id.get(i));
			// citys[cid_map.get(o_c_id2.get(i))-1].add(city.get(i));
			cdids[cid_map.get(o_c_id2.get(i)) - 1].add(o_d_id.get(i));
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
		numo = 1;
		int cnt = 1;
		int[] a = new int[6]; // 不同区域的订单id
		for (int i = 0; i < 6; i++) {
			a[i] = 1;
		}
		oorder.oo_d_id.add(o_d_id.get(0));
		// 初始区域id
		oorder.oo_id.add(a[o_d_id.get(0) - 1]++);
		oorder.oo_c_id.add(cid_map.get(o_c_id2.get(0)));
		for (int i = 1; i < numol; i++) {
			if (o_id.get(i).equals(o_id.get(i - 1))) {
				// System.out.println(80);
				cnt++;
				continue;
			}
			oorder.oo_d_id.add(o_d_id.get(i));// 区域id
			numo++;
			// oorder.oo_id.add(a[o_d_id.get(i)-1]++);
			if (o_d_id.get(i) == 1) { // 订单id
				oorder.oo_id.add(a[0]++);
			} else if (o_d_id.get(i) == 2) {
				oorder.oo_id.add(a[1]++);
			} else if (o_d_id.get(i) == 3) {
				oorder.oo_id.add(a[2]++);
			} else if (o_d_id.get(i) == 4) {
				oorder.oo_id.add(a[3]++);
			} else if (o_d_id.get(i) == 5) {
				oorder.oo_id.add(a[4]++);
			} else {
				oorder.oo_id.add(a[5]++);
			}
			oorder.oo_c_id.add(cid_map.get(o_c_id2.get(i))); // 客户id
			oorder.oo_ol_cnt.add(cnt); // 订单项目数量
			cnt = 1;
		}
		oorder.oo_ol_cnt.add(cnt);
		// for(int j:oorder.oo_c_id) {
		// System.out.println(j);
		// }
		// System.out.println(numo);
		// Collections.sort(oorder.oo_d_id);

		// 取商品id后8位，排序，赋值（相同内容值一样），还原
		List<String> o_i_id1 = new ArrayList<>();
		List<String> o_i_id2 = new ArrayList<>();
		HashMap<String, Integer> iid_map = new HashMap<>();
		// o_c_id1 = o_c_id;
		for (int i = 0; i < numol; i++) {
			o_i_id1.add(o_item_id.get(i));
			o_i_id2.add(o_item_id.get(i));
		}
		Collections.sort(o_i_id1);
		numi = 0;
		iid_map.put(o_i_id1.get(0), ++numi); // map中的键值对是按顺序插入的，key值是商品id后8位
		for (int i = 1; i < numol; i++) { // 商品id
			if (o_i_id1.get(i).equals(o_i_id1.get(i - 1))) {
				iid_map.put(o_i_id1.get(i), numi);
			} else {
				iid_map.put(o_i_id1.get(i), ++numi);
			}
		}

		// for(int i=0;i<numol;i++)
		// {
		// String filePath = "/opt/loadDataWorker.txt";
		// try (PrintWriter pw = new PrintWriter(new FileWriter(filePath, true))) {
		// pw.print(o_c_id.get(i) + " ");
		// pw.print(cid_map.get(o_c_id.get(i)) + " \t");
		// pw.print(o_item_id.get(i) + " ");
		// pw.println(iid_map.get(o_item_id.get(i)));
		// } catch (IOException e) {
		// e.printStackTrace();
		// }
		// }

		// orderline订单表插入的内容
		orderline = new orderlinedata();
		int olnumber = 1;
		for (int i = 0; i < 6; i++) {
			a[i] = 0;
		}
		// orderline.ol_number.add(olnumber);
		for (int i = 0; i < numol; i++) {
			orderline.ol_d_id.add(o_d_id.get(i));
			// orderline.ol_o_id.add(a[o_d_id.get(i)-1]++);
			if (i >= 1 && o_id.get(i).equals(o_id.get(i - 1))) {
				orderline.ol_o_id.add(a[o_d_id.get(i) - 1]);
				orderline.ol_number.add(++olnumber);
			} else {
				olnumber = 1;
				orderline.ol_o_id.add(++a[o_d_id.get(i) - 1]);
				orderline.ol_number.add(olnumber);
			}
			orderline.ol_i_id.add(iid_map.get(o_i_id2.get(i)));
			orderline.ol_amount.add(Double.parseDouble(amount.get(i)));
			orderline.ol_quantity.add(Integer.parseInt(quantity.get(i)));
			orderline.ol_data.add(oldata.get(i));
		}
		// for(int j:orderline.ol_o_id) {
		// System.out.println(j);
		// }

		item = new itemdata();
		HashSet<String> itid = new HashSet<>();
		for (int i = 0; i < numol; i++) {
			if (itid.contains(o_item_id.get(i))) {
				continue;
			}
			itid.add(o_item_id.get(i));
			item.i_id.add(iid_map.get(o_i_id2.get(i)));
			item.i_name.add(iname.get(i));
			item.i_price.add(Double.parseDouble(amount.get(i)) / Integer.parseInt(quantity.get(i)));
			item.i_data.add(idata.get(i));
		}

		district = new districtdata();
		double[] dnum = new double[numd];
		for (int i = 0; i < numd; i++) {
			dnum[i] = 0;
		}
		for (int i = 0; i < numol; i++) {
			dnum[o_d_id.get(i) - 1] += Double.parseDouble(amount.get(i));
		}
		double wytd = 0;
		for (int i = 0; i < numd; i++) {
			// district.d_name.add();
			district.d_ytd.add(dnum[i]);
			wytd = wytd + dnum[i];
			district.d_name.add(dmap2.get(i + 1));
			district.d_next_o_id.add(a[i] + 1);
			district.d_city.add(dcitys.get(i));
			district.d_state.add(dstates.get(i));
		}
		warehouse = new warehousedata();
		warehouse.w_ytd.add(wytd);
	}

	/*
	 * run()
	 */
	public void run() {
		int job;
		try {
			while ((job = LoadData.getNextJob()) >= 0) {
				if (job == 0) {
					fmt.format("Worker %03d: Loading ITEM", worker);
					System.out.println(sb.toString());
					sb.setLength(0);

					loadItem();

					fmt.format("Worker %03d: Loading ITEM done", worker);
					System.out.println(sb.toString());
					sb.setLength(0);
				} else {
					fmt.format("Worker %03d: Loading Warehouse %6d", worker, job);
					System.out.println(sb.toString());
					sb.setLength(0);

					loadWarehouse(job);

					fmt.format("Worker %03d: Loading Warehouse %6d done", worker, job);
					System.out.println(sb.toString());
					sb.setLength(0);
				}
			}

			/*
			 * Close the DB connection if in direct DB mode.
			 */
			if (!writeCSV)
				dbConn.close();
		} catch (SQLException se) {
			while (se != null) {
				fmt.format("Worker %03d: ERROR: %s", worker, se.getMessage());
				System.err.println(sb.toString());
				sb.setLength(0);
				se = se.getNextException();
			}
		} catch (Exception e) {
			fmt.format("Worker %03d: ERROR: %s", worker, e.getMessage());
			System.err.println(sb.toString());
			sb.setLength(0);
			e.printStackTrace();
			return;
		}
	} // End run()

	/*
	 * ----
	 * loadItem()
	 *
	 * Load the content of the ITEM table.
	 * ----
	 */
	private void loadItem() throws SQLException, IOException {
		int i_id;

		if (writeCSV) {
			/*
			 * Saving CONFIG information in CSV mode.
			 */
			fmtConfig.format("warehouses,%d\n", LoadData.getNumWarehouses());
			fmtConfig.format("nURandCLast,%d\n", rnd.getNURandCLast());
			fmtConfig.format("nURandCC_ID,%d\n", rnd.getNURandCC_ID());
			fmtConfig.format("nURandCI_ID,%d\n", rnd.getNURandCI_ID());
			LoadData.configAppend(sbConfig);
		} else {
			/*
			 * Saving CONFIG information in DB mode.
			 */
			stmtConfig.setString(1, "warehouses");
			stmtConfig.setString(2, "" + LoadData.getNumWarehouses());
			stmtConfig.execute();

			stmtConfig.setString(1, "nURandCLast");
			stmtConfig.setString(2, "" + rnd.getNURandCLast());
			stmtConfig.execute();

			stmtConfig.setString(1, "nURandCC_ID");
			stmtConfig.setString(2, "" + rnd.getNURandCC_ID());
			stmtConfig.execute();

			stmtConfig.setString(1, "nURandCI_ID");
			stmtConfig.setString(2, "" + rnd.getNURandCI_ID());
			stmtConfig.execute();
		}

		/*
		 * for (i_id = 1; i_id <= 100000; i_id++)
		 * {
		 * String iData;
		 * 
		 * if (i_id != 1 && (i_id - 1) % 1000 == 0)
		 * {
		 * if (writeCSV)
		 * {
		 * LoadData.itemAppend(sbItem);
		 * }
		 * else
		 * {
		 * stmtItem.executeBatch();
		 * stmtItem.clearBatch();
		 * }
		 * }
		 * 
		 * // Clause 4.3.3.1 for ITEM
		 * if (rnd.nextInt(1, 100) <= 10)
		 * {
		 * int len = rnd.nextInt(26, 50);
		 * int off = rnd.nextInt(0, len - 8);
		 * 
		 * iData = rnd.getAString(off, off) +
		 * "ORIGINAL" +
		 * rnd.getAString(len - off - 8, len - off - 8);
		 * }
		 * else
		 * {
		 * iData = rnd.getAString(26, 50);
		 * }
		 * 
		 * if (writeCSV)
		 * {
		 * fmtItem.format("%d,%s,%.2f,%s,%d\n",
		 * i_id,
		 * rnd.getAString(14, 24),
		 * ((double)rnd.nextLong(100, 10000)) / 100.0,
		 * iData,
		 * rnd.nextInt(1, 10000));
		 * 
		 * }
		 * else
		 * {
		 * stmtItem.setInt(1, i_id);
		 * stmtItem.setInt(2, rnd.nextInt(1, 10000));
		 * stmtItem.setString(3, rnd.getAString(14, 24));
		 * stmtItem.setDouble(4, ((double)rnd.nextLong(100, 10000)) / 100.0);
		 * stmtItem.setString(5, iData);
		 * 
		 * stmtItem.addBatch();
		 * }
		 * }
		 */
		for (int i = 0; i < numi; i++) {
			stmtItem.setInt(1, item.i_id.get(i));
			stmtItem.setInt(2, rnd.nextInt(1, 10000));
			stmtItem.setString(3, item.i_name.get(i));
			stmtItem.setDouble(4, item.i_price.get(i));
			stmtItem.setString(5, item.i_data.get(i));
			stmtItem.addBatch();
		}

		if (writeCSV) {
			LoadData.itemAppend(sbItem);
		} else {
			stmtItem.executeBatch();
			stmtItem.clearBatch();
			// stmtItem.close();
			dbConn.commit();
		}

	} // End loadItem()

	/*
	 * ----
	 * loadWarehouse()
	 *
	 * Load the content of one warehouse.
	 * ----
	 */
	private void loadWarehouse(int w_id) throws SQLException, IOException {
		/*
		 * Load the WAREHOUSE row.
		 */
		if (writeCSV) {
			fmtWarehouse.format("%d,%.2f,%.4f,%s,%s,%s,%s,%s,%s\n", w_id,
					300000.0,
					((double) rnd.nextLong(0, 2000)) / 10000.0,
					rnd.getAString(6, 10),
					rnd.getAString(10, 20),
					rnd.getAString(10, 20),
					rnd.getAString(10, 20),
					rnd.getState(),
					rnd.getNString(4, 4) + "11111");
			LoadData.warehouseAppend(sbWarehouse);
		} else {
			stmtWarehouse.setInt(1, w_id);
			stmtWarehouse.setString(2, "仓库1"); // rnd.getAString(6, 10));
			stmtWarehouse.setString(3, "无"); // rnd.getAString(10, 20));
			stmtWarehouse.setString(4, "无"); // rnd.getAString(10, 20));
			stmtWarehouse.setString(5, "沈阳"); // rnd.getAString(10, 20));
			stmtWarehouse.setString(6, "辽宁"); // rnd.getState());
			stmtWarehouse.setString(7, "110000"); // rnd.getNString(4, 4) + "11111");
			stmtWarehouse.setDouble(8, ((double) rnd.nextLong(0, 2000)) / 10000.0);
			stmtWarehouse.setDouble(9, warehouse.w_ytd.get(0));
			stmtWarehouse.execute();
		}

		/*
		 * For each WAREHOUSE there are 100,000 STOCK rows.
		 */
		/*
		 * for (int s_i_id = 1; s_i_id <= 100000; s_i_id++)
		 * {
		 * String sData;
		 */
		/*
		 * Load the data in batches of 10,000 rows.
		 */
		/*
		 * if (s_i_id != 1 && (s_i_id - 1) % 10000 == 0)
		 * {
		 * if (writeCSV)
		 * LoadData.warehouseAppend(sbWarehouse);
		 * else
		 * {
		 * stmtStock.executeBatch();
		 * stmtStock.clearBatch();
		 * }
		 * }
		 * 
		 * // Clause 4.3.3.1 for STOCK
		 * if (rnd.nextInt(1, 100) <= 10)
		 * {
		 * int len = rnd.nextInt(26, 50);
		 * int off = rnd.nextInt(0, len - 8);
		 * 
		 * sData = rnd.getAString(off, off) +
		 * "ORIGINAL" +
		 * rnd.getAString(len - off - 8, len - off - 8);
		 * }
		 * else
		 * {
		 * sData = rnd.getAString(26, 50);
		 * }
		 * 
		 * if (writeCSV)
		 * {
		 * fmtStock.format("%d,%d,%d,%d,%d,%d,%s," +
		 * "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
		 * s_i_id,
		 * w_id,
		 * rnd.nextInt(10, 100),
		 * 0,
		 * 0,
		 * 0,
		 * sData,
		 * rnd.getAString(24, 24),
		 * rnd.getAString(24, 24),
		 * rnd.getAString(24, 24),
		 * rnd.getAString(24, 24),
		 * rnd.getAString(24, 24),
		 * rnd.getAString(24, 24),
		 * rnd.getAString(24, 24),
		 * rnd.getAString(24, 24),
		 * rnd.getAString(24, 24),
		 * rnd.getAString(24, 24));
		 * }
		 * else
		 * {
		 * stmtStock.setInt(1, s_i_id);
		 * stmtStock.setInt(2, w_id);
		 * stmtStock.setInt(3, rnd.nextInt(10, 100));
		 * stmtStock.setString(4, rnd.getAString(24, 24));
		 * stmtStock.setString(5, rnd.getAString(24, 24));
		 * stmtStock.setString(6, rnd.getAString(24, 24));
		 * stmtStock.setString(7, rnd.getAString(24, 24));
		 * stmtStock.setString(8, rnd.getAString(24, 24));
		 * stmtStock.setString(9, rnd.getAString(24, 24));
		 * stmtStock.setString(10, rnd.getAString(24, 24));
		 * stmtStock.setString(11, rnd.getAString(24, 24));
		 * stmtStock.setString(12, rnd.getAString(24, 24));
		 * stmtStock.setString(13, rnd.getAString(24, 24));
		 * stmtStock.setInt(14, 0);
		 * stmtStock.setInt(15, 0);
		 * stmtStock.setInt(16, 0);
		 * stmtStock.setString(17, sData);
		 * 
		 * stmtStock.addBatch();
		 * }
		 * 
		 * }
		 */
		for (int i = 0; i < numi; i++) {
			stmtStock.setInt(1, item.i_id.get(i));
			stmtStock.setInt(2, 1);
			stmtStock.setInt(3, rnd.nextInt(10, 100));
			stmtStock.setString(4, "1");// rnd.getAString(24, 24));
			stmtStock.setString(5, "2"); // rnd.getAString(24, 24));
			stmtStock.setString(6, "3");// rnd.getAString(24, 24));
			stmtStock.setString(7, "4");// rnd.getAString(24, 24));
			stmtStock.setString(8, "5");// rnd.getAString(24, 24));
			stmtStock.setString(9, "6");// rnd.getAString(24, 24));
			stmtStock.setString(10, "7");// rnd.getAString(24, 24));
			stmtStock.setString(11, "8");// rnd.getAString(24, 24));
			stmtStock.setString(12, "9");// rnd.getAString(24, 24));
			stmtStock.setString(13, "10");// rnd.getAString(24, 24));
			stmtStock.setInt(14, 0);
			stmtStock.setInt(15, 0);
			stmtStock.setInt(16, 0);
			stmtStock.setString(17, "未知");
			stmtStock.addBatch();
		}
		if (writeCSV) {
			LoadData.stockAppend(sbStock);
		} else {
			stmtStock.executeBatch();
			stmtStock.clearBatch();
		}

		/*
		 * For each WAREHOUSE there are 10 DISTRICT rows.
		 */
		for (int d_id = 1; d_id <= numd; d_id++)// 10
		{
			if (writeCSV) {
				fmtDistrict.format("%d,%d,%.2f,%.4f,%d,%s,%s,%s,%s,%s,%s\n",
						d_id,
						w_id,
						30000.0,
						((double) rnd.nextLong(0, 2000)) / 10000.0,
						3001,
						rnd.getAString(6, 10),
						rnd.getAString(10, 20),
						rnd.getAString(10, 20),
						rnd.getAString(10, 20),
						rnd.getState(),
						rnd.getNString(4, 4) + "11111");

				LoadData.districtAppend(sbDistrict);
			} else {
				stmtDistrict.setInt(1, d_id);
				stmtDistrict.setInt(2, w_id);
				stmtDistrict.setString(3, district.d_name.get(d_id - 1));
				stmtDistrict.setString(4, "地址1");// rnd.getAString(10, 20));
				stmtDistrict.setString(5, "地址2");
				stmtDistrict.setString(6, district.d_city.get(d_id - 1));
				stmtDistrict.setString(7, district.d_state.get(d_id - 1));
				stmtDistrict.setString(8, rnd.getNString(6, 6));
				stmtDistrict.setDouble(9, ((double) rnd.nextLong(0, 2000)) / 10000.0);
				stmtDistrict.setDouble(10, district.d_ytd.get(d_id - 1));// d_id从1-6
				stmtDistrict.setInt(11, district.d_next_o_id.get(d_id - 1));

				stmtDistrict.execute();
			}

			/*
			 * Within each DISTRICT there are 3,000 CUSTOMERs.
			 */
			/*
			 * for (int c_id = 1; c_id <= 3000; c_id++)
			 * {
			 * if (writeCSV)
			 * {
			 * fmtCustomer.format("%d,%d,%d,%.4f,%s,%s,%s," +
			 * "%.2f,%.2f,%.2f,%d,%d," +
			 * "%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
			 * c_id,
			 * d_id,
			 * w_id,
			 * ((double)rnd.nextLong(0, 5000)) / 10000.0,
			 * (rnd.nextInt(1, 100) <= 90) ? "GC" : "BC",
			 * (c_id <= 1000) ? rnd.getCLast(c_id - 1) : rnd.getCLast(),
			 * rnd.getAString(8, 16),
			 * 50000.00,
			 * -10.00,
			 * 10.00,
			 * 1,
			 * 0,
			 * rnd.getAString(10, 20),
			 * rnd.getAString(10, 20),
			 * rnd.getAString(10, 20),
			 * rnd.getState(),
			 * rnd.getNString(4, 4) + "11111",
			 * rnd.getNString(16, 16),
			 * new java.sql.Timestamp(System.currentTimeMillis()).toString(),
			 * "OE",
			 * rnd.getAString(300, 500));
			 * }
			 * else
			 * {
			 * stmtCustomer.setInt(1, c_id);
			 * stmtCustomer.setInt(2, d_id);
			 * stmtCustomer.setInt(3, w_id);
			 * stmtCustomer.setString(4, rnd.getAString(8, 16));
			 * stmtCustomer.setString(5, "OE");
			 * if (c_id <= 1000)
			 * stmtCustomer.setString(6, rnd.getCLast(c_id - 1));
			 * else
			 * stmtCustomer.setString(6, rnd.getCLast());
			 * stmtCustomer.setString(7, rnd.getAString(10, 20));
			 * stmtCustomer.setString(8, rnd.getAString(10, 20));
			 * stmtCustomer.setString(9, rnd.getAString(10, 20));
			 * stmtCustomer.setString(10, rnd.getState());
			 * stmtCustomer.setString(11, rnd.getNString(6, 6));
			 * stmtCustomer.setString(12, rnd.getNString(11, 11));
			 * stmtCustomer.setTimestamp(13, new
			 * java.sql.Timestamp(System.currentTimeMillis()));
			 * if (rnd.nextInt(1, 100) <= 90)
			 * stmtCustomer.setString(14, "GC");
			 * else
			 * stmtCustomer.setString(14, "BC");
			 * stmtCustomer.setDouble(15, 50000.00);
			 * stmtCustomer.setDouble(16, ((double)rnd.nextLong(0, 5000)) / 10000.0);
			 * stmtCustomer.setDouble(17, -10.00);
			 * stmtCustomer.setDouble(18, 10.00);
			 * stmtCustomer.setInt(19, 1);
			 * stmtCustomer.setInt(20, 1);
			 * stmtCustomer.setString(21, rnd.getAString(300, 500));
			 * 
			 * stmtCustomer.addBatch();
			 * }
			 */

			/*
			 * For each CUSTOMER there is one row in HISTORY.
			 */
			/*
			 * if (writeCSV)
			 * {
			 * fmtHistory.format("%d,%d,%d,%d,%d,%d,%s,%.2f,%s\n",
			 * (w_id - 1) * 30000 + (d_id - 1) * 3000 + c_id,
			 * c_id,
			 * d_id,
			 * w_id,
			 * d_id,
			 * w_id,
			 * new java.sql.Timestamp(System.currentTimeMillis()).toString(),
			 * 10.00,
			 * rnd.getAString(12, 24));
			 * }
			 * else
			 * {
			 * stmtHistory.setInt(1, (w_id - 1) * 30000 + (d_id - 1) * 3000 + c_id);
			 * stmtHistory.setInt(2, c_id);
			 * stmtHistory.setInt(3, d_id);
			 * stmtHistory.setInt(4, w_id);
			 * stmtHistory.setInt(5, d_id);
			 * stmtHistory.setInt(6, w_id);
			 * stmtHistory.setTimestamp(7, new
			 * java.sql.Timestamp(System.currentTimeMillis()));
			 * stmtHistory.setDouble(8, 10.00);
			 * stmtHistory.setString(9, rnd.getAString(12, 24));
			 * 
			 * stmtHistory.addBatch();
			 * }
			 * }
			 * 
			 * if (writeCSV)
			 * {
			 * LoadData.customerAppend(sbCustomer);
			 * LoadData.historyAppend(sbHistory);
			 * }
			 * else
			 * {
			 * stmtCustomer.executeBatch();
			 * stmtCustomer.clearBatch();
			 * stmtHistory.executeBatch();
			 * stmtHistory.clearBatch();
			 * }
			 */

			/*
			 * For the ORDER rows the TPC-C specification demands that they
			 * are generated using a random permutation of all 3,000
			 * customers. To do that we set up an array with all C_IDs
			 * and then randomly shuffle it.
			 */
			/*
			 * int randomCID[] = new int[3000];
			 * for (int i = 0; i < 3000; i++)
			 * randomCID[i] = i + 1;
			 * for (int i = 0; i < 3000; i++)
			 * {
			 * int x = rnd.nextInt(0, 2999);
			 * int y = rnd.nextInt(0, 2999);
			 * int tmp = randomCID[x];
			 * randomCID[x] = randomCID[y];
			 * randomCID[y] = tmp;
			 * }
			 * 
			 * for (int o_id = 1; o_id <= 3000; o_id++)
			 * {
			 * int o_ol_cnt = rnd.nextInt(5, 15);
			 * 
			 * if (writeCSV)
			 * {
			 * fmtOrder.format("%d,%d,%d,%d,%s,%d,%d,%s\n",
			 * o_id,
			 * w_id,
			 * d_id,
			 * randomCID[o_id - 1],
			 * (o_id < 2101) ? rnd.nextInt(1, 10) : csvNull,
			 * o_ol_cnt,
			 * 1,
			 * new java.sql.Timestamp(System.currentTimeMillis()).toString());
			 * }
			 * else
			 * {
			 * stmtOrder.setInt(1, o_id);
			 * stmtOrder.setInt(2, d_id);
			 * stmtOrder.setInt(3, w_id);
			 * stmtOrder.setInt(4, randomCID[o_id - 1]);
			 * stmtOrder.setTimestamp(5, new
			 * java.sql.Timestamp(System.currentTimeMillis()));
			 * if (o_id < 2101)
			 * stmtOrder.setInt(6, rnd.nextInt(1, 10));
			 * else
			 * stmtOrder.setNull(6, java.sql.Types.INTEGER);
			 * stmtOrder.setInt(7, o_ol_cnt);
			 * stmtOrder.setInt(8, 1);
			 * 
			 * stmtOrder.addBatch();
			 * }
			 */

			/*
			 * Create the ORDER_LINE rows for this ORDER.
			 */
			/*
			 * for (int ol_number = 1; ol_number <= o_ol_cnt; ol_number++)
			 * {
			 * long now = System.currentTimeMillis();
			 * 
			 * if (writeCSV)
			 * {
			 * fmtOrderLine.format("%d,%d,%d,%d,%d,%s,%.2f,%d,%d,%s\n",
			 * w_id,
			 * d_id,
			 * o_id,
			 * ol_number,
			 * rnd.nextInt(1, 100000),
			 * (o_id < 2101) ? new java.sql.Timestamp(now).toString() : csvNull,
			 * (o_id < 2101) ? 0.00 : ((double)rnd.nextLong(1, 999999)) / 100.0,
			 * w_id,
			 * 5,
			 * rnd.getAString(24, 24));
			 * }
			 * else
			 * {
			 * stmtOrderLine.setInt(1, o_id);
			 * stmtOrderLine.setInt(2, d_id);
			 * stmtOrderLine.setInt(3, w_id);
			 * stmtOrderLine.setInt(4, ol_number);
			 * stmtOrderLine.setInt(5, rnd.nextInt(1, 100000));
			 * stmtOrderLine.setInt(6, w_id);
			 * if (o_id < 2101)
			 * stmtOrderLine.setTimestamp(7, new java.sql.Timestamp(now));
			 * else
			 * stmtOrderLine.setNull(7, java.sql.Types.TIMESTAMP);
			 * stmtOrderLine.setInt(8, 5);
			 * if (o_id < 2101)
			 * stmtOrderLine.setDouble(9, 0.00);
			 * else
			 * stmtOrderLine.setDouble(9, ((double)rnd.nextLong(1, 999999)) / 100.0);
			 * stmtOrderLine.setString(10, rnd.getAString(24, 24));
			 * 
			 * stmtOrderLine.addBatch();
			 * }
			 * }
			 */

			/*
			 * The last 900 ORDERs are not yet delieverd and have a
			 * row in NEW_ORDER.
			 */
			/*
			 * if (o_id >= 2101)
			 * {
			 * if (writeCSV)
			 * {
			 * fmtNewOrder.format("%d,%d,%d\n",
			 * w_id,
			 * d_id,
			 * o_id);
			 * }
			 * else
			 * {
			 * stmtNewOrder.setInt(1, o_id);
			 * stmtNewOrder.setInt(2, d_id);
			 * stmtNewOrder.setInt(3, w_id);
			 * 
			 * stmtNewOrder.addBatch();
			 * }
			 * }
			 * }
			 */

		}
		for (int i = 0; i < numc; i++) {
			stmtCustomer.setInt(1, customer.c_id.get(i));
			stmtCustomer.setInt(2, customer.c_d_id.get(i));
			stmtCustomer.setInt(3, w_id);
			stmtCustomer.setString(4, customer.c_first.get(i));
			stmtCustomer.setString(5, " ");
			// if (c_id <= 1000)
			// stmtCustomer.setString(6, rnd.getCLast(c_id - 1));
			// else
			stmtCustomer.setString(6, customer.c_last.get(i));
			stmtCustomer.setString(7, "地址1");// rnd.getAString(10, 20));
			stmtCustomer.setString(8, "地址2");// rnd.getAString(10, 20));
			stmtCustomer.setString(9, customer.c_city.get(i));
			stmtCustomer.setString(10, customer.c_state.get(i));
			stmtCustomer.setString(11, rnd.getNString(6, 6));
			stmtCustomer.setString(12, rnd.getNString(11, 11));
			stmtCustomer.setTimestamp(13, new java.sql.Timestamp(System.currentTimeMillis()));
			if (rnd.nextInt(1, 100) <= 90)
				stmtCustomer.setString(14, "GC");
			else
				stmtCustomer.setString(14, "BC");
			stmtCustomer.setDouble(15, 50000.00);
			stmtCustomer.setDouble(16, customer.c_discount.get(i));
			stmtCustomer.setDouble(17, 1000.00);
			stmtCustomer.setDouble(18, 10.00);
			stmtCustomer.setInt(19, 1);
			stmtCustomer.setInt(20, 1);
			stmtCustomer.setString(21, "暂无"); // rnd.getAString(300, 500));
			stmtCustomer.addBatch();

			// 历史表,每个客户有一行
			stmtHistory.setInt(1, i); // (w_id - 1) * 30000 + (d_id - 1) * 3000 + c_id);
			stmtHistory.setInt(2, customer.c_id.get(i));
			stmtHistory.setInt(3, customer.c_d_id.get(i));
			stmtHistory.setInt(4, w_id);
			stmtHistory.setInt(5, customer.c_d_id.get(i));
			stmtHistory.setInt(6, w_id);
			stmtHistory.setTimestamp(7, new java.sql.Timestamp(System.currentTimeMillis()));
			stmtHistory.setDouble(8, 10.00);
			stmtHistory.setString(9, "暂无"); // rnd.getAString(12, 24));
			stmtHistory.addBatch();
		}
		if (writeCSV) {
			LoadData.customerAppend(sbCustomer);
			LoadData.historyAppend(sbHistory);
		} else {
			stmtCustomer.executeBatch();
			stmtCustomer.clearBatch();
			stmtHistory.executeBatch();
			stmtHistory.clearBatch();
		}
		for (int i = 0; i < numo; i++) {
			stmtOrder.setInt(1, oorder.oo_id.get(i));
			stmtOrder.setInt(2, oorder.oo_d_id.get(i));
			stmtOrder.setInt(3, w_id);
			stmtOrder.setInt(4, oorder.oo_c_id.get(i));
			stmtOrder.setTimestamp(5, new java.sql.Timestamp(System.currentTimeMillis()));
			// if (o_id < 2101)
			stmtOrder.setInt(6, rnd.nextInt(1, 10));
			// else
			// stmtOrder.setNull(6, java.sql.Types.INTEGER);
			stmtOrder.setInt(7, oorder.oo_ol_cnt.get(i));
			stmtOrder.setInt(8, 1);

			stmtOrder.addBatch();
			if (oorder.oo_id.get(i) > 50) {
				stmtNewOrder.setInt(1, oorder.oo_id.get(i));
				stmtNewOrder.setInt(2, oorder.oo_d_id.get(i));
				stmtNewOrder.setInt(3, w_id);
				stmtNewOrder.addBatch();
			}
		}
		for (int i = 0; i < numol; i++) {
			stmtOrderLine.setInt(1, orderline.ol_o_id.get(i));
			stmtOrderLine.setInt(2, orderline.ol_d_id.get(i));
			stmtOrderLine.setInt(3, w_id);
			stmtOrderLine.setInt(4, orderline.ol_number.get(i));
			stmtOrderLine.setInt(5, orderline.ol_i_id.get(i));
			stmtOrderLine.setInt(6, w_id);
			// if (o_id < 2101)
			stmtOrderLine.setTimestamp(7, new java.sql.Timestamp(System.currentTimeMillis()));
			// else
			// stmtOrderLine.setNull(7, java.sql.Types.TIMESTAMP);
			stmtOrderLine.setInt(8, orderline.ol_quantity.get(i));
			// if (o_id < 2101)
			stmtOrderLine.setDouble(9, orderline.ol_amount.get(i));
			// else
			// stmtOrderLine.setDouble(9, ((double)rnd.nextLong(1, 999999)) / 100.0);
			stmtOrderLine.setString(10, orderline.ol_data.get(i));
			stmtOrderLine.addBatch();
		}

		if (writeCSV) {
			LoadData.orderAppend(sbOrder);
			LoadData.orderLineAppend(sbOrderLine);
			LoadData.newOrderAppend(sbNewOrder);
		} else {
			stmtOrder.executeBatch();
			stmtOrder.clearBatch();
			stmtOrderLine.executeBatch();
			stmtOrderLine.clearBatch();
			stmtNewOrder.executeBatch();
			stmtNewOrder.clearBatch();
		}

		if (!writeCSV)
			dbConn.commit();
	} // End loadWarehouse()
}
