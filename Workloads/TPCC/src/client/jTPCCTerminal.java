
/*
 * jTPCCTerminal - Terminal emulator code for jTPCC (transactions)
 *
 * Copyright (C) 2003, Raul Barbosa
 * Copyright (C) 2004-2016, Denis Lussier
 * Copyright (C) 2016, Jan Wieck
 *
 */
import org.apache.log4j.*;
import java.io.*;
import java.sql.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

import javax.swing.*;

public class jTPCCTerminal implements jTPCCConfig, Runnable {
	private String concurrencyControl = "none"; // #RAIN
	private static org.apache.log4j.Logger log = Logger.getLogger(jTPCCTerminal.class);
	private String terminalName;
	private Connection conn = null;
	private Statement stmt = null;
	private Statement stmt1 = null;
	private ResultSet rs = null;
	private int terminalWarehouseID, terminalDistrictID;
	private boolean terminalWarehouseFixed;
	private int newOrderWeight, paymentWeight, orderStatusWeight, deliveryWeight, stockLevelWeight, limPerMin_Terminal;
	private jTPCC parent;
	private jTPCCRandom rnd;

	private jTPCCRandom[] rnd1;
	private int seed;
	private int[] same;
	Loadcsv csv = null;
	private static Object lock = new Object();

	private int transactionCount = 1;
	private int numTransactions;
	private int numWarehouses;
	private int newOrderCounter;
	private long totalTnxs = 1;
	private StringBuffer query = null;
	private int result = 0;
	private boolean stopRunningSignal = false;

	private boolean isWaiting = false; // #RAIN
	private boolean finishNewOrder = false; // #RAIN
	private int reservedOOid = -1; // #RAIN
	private int retryCount = 0; // #RAIN
	private int terminalId = 0; // #RAIN
	private List<Integer> waitedTxns; // #RAIN
	private Object txnLock; // #RAIN
	private Object listLock; // #RAIN
	private Object waitLock; // #RAIN

	long terminalStartTime = 0;
	long terminalEndTime = 0;

	jTPCCConnection db = null;
	int dbType = 0;

	/* ---------------------- #RAIN ---------------------- */

	public Object getTxnLock() {
		return txnLock;
	}

	public Object getListLock() {
		return listLock;
	}

	public Object getWaitLock() {
		return listLock;
	}

	public void setWaiting(boolean waiting_status) {
		synchronized (waitLock) {
			isWaiting = waiting_status;
		}
	}

	public boolean checkWaiting() {
		synchronized (waitLock) {
			return isWaiting;
		}
	}

	public List<Integer> getWaitedTxns() {
		synchronized (listLock) {
			return waitedTxns;
		}
	}

	public void addToWaitedTxns(Integer waited_ooid) {
		synchronized (listLock) {
			waitedTxns.add(waited_ooid);
		}
	}

	public void clearWaitedTxns() {
		synchronized (listLock) {
			waitedTxns.clear();
		}
	}

	/* ---------------------- #RAIN ---------------------- */

	public jTPCCTerminal(String terminalName, int terminalWarehouseID, int terminalDistrictID, Connection conn,
			int dbType, int numTransactions, boolean terminalWarehouseFixed, int newOrderWeight, int paymentWeight,
			int orderStatusWeight,
			int deliveryWeight, int stockLevelWeight, int numWarehouses, int limPerMin_Terminal,
			int i, jTPCC parent, Loadcsv csv, String concurrencyControl, int terminalId) throws SQLException {
		this.terminalName = terminalName;
		this.conn = conn;
		this.dbType = dbType;
		this.stmt = conn.createStatement();
		this.stmt.setMaxRows(200);
		this.stmt.setFetchSize(100);

		this.stmt1 = conn.createStatement();
		this.stmt1.setMaxRows(1);

		this.terminalWarehouseID = terminalWarehouseID;
		this.terminalDistrictID = terminalDistrictID;
		this.terminalWarehouseFixed = terminalWarehouseFixed;
		this.parent = parent;
		this.rnd = parent.getRnd().newRandom();
		this.numTransactions = numTransactions;
		this.newOrderWeight = newOrderWeight;
		this.paymentWeight = paymentWeight;
		this.orderStatusWeight = orderStatusWeight;
		this.deliveryWeight = deliveryWeight;
		this.stockLevelWeight = stockLevelWeight;
		this.numWarehouses = numWarehouses;
		this.newOrderCounter = 0;
		this.limPerMin_Terminal = limPerMin_Terminal;
		this.db = new jTPCCConnection(conn, dbType);
		this.csv = csv;
		this.seed = i;
		this.concurrencyControl = concurrencyControl;
		this.terminalId = terminalId;
		this.waitedTxns = new ArrayList<>();

		txnLock = new Object(); // #RAIN
		listLock = new Object(); // #RAIN
		waitLock = new Object(); // #RAIN

		if (numTransactions != -1) {
			this.rnd1 = new jTPCCRandom[numTransactions];
			this.same = new int[numTransactions];
			HashSet<Integer> used = new HashSet<>();
			for (int j = 0; j < numTransactions * 0.8; j++) {
				int seed1;
				while (true) {
					if ((numTransactions * 0.8) % 1 == 0) {
						seed1 = rnd.nextInt(seed * numTransactions,
								(int) Math.floor(seed * numTransactions + numTransactions * 0.8) - 1);
					} else {
						seed1 = rnd.nextInt(seed * numTransactions,
								(int) Math.floor(seed * numTransactions + numTransactions * 0.8));
					}
					if (!used.contains(seed1)) {
						used.add(seed1);
						break;
					}
				}
				// this.rnd1[j] =parent.getRnd().newRandom(seed1);
				this.rnd1[j] = new jTPCCRandom(seed1);
			}
			for (int j = 0; j < numTransactions; j++) {
				this.same[j] = rnd.nextInt(0, 100);
			}
		}

		terminalMessage("");
		terminalMessage("Terminal \'" + terminalName + "\' has WarehouseID=" + terminalWarehouseID + " and DistrictID="
				+ terminalDistrictID + ".");
	}

	public void run() {
		if (terminalStartTime == 0) {
			terminalStartTime = System.currentTimeMillis();
		}

		if (concurrencyControl.equals("aria")) {
			executeTransactions(1);
		} else if (concurrencyControl.equals("calvin")) {
			try {
				Thread.sleep(20); // Calvin需要预先读快照，因此设置延迟
			} catch (InterruptedException e) {
				System.out.println(e);
			}
			executeTransactions(1);
		} else {
			executeTransactions(numTransactions);
		}

		if ((!concurrencyControl.equals("aria") && !concurrencyControl.equals("calvin")
				&& !concurrencyControl.equals("s2pl")) || (parent.terminalsClose() && reservedOOid == -1)) {
			if (terminalEndTime == 0) {
				terminalEndTime = System.currentTimeMillis();
				parent.addToExecTime(terminalEndTime - terminalStartTime);
			}

			try {
				printMessage("");
				printMessage("Closing statement and connection...");

				stmt.close();
				conn.close();
			} catch (Exception e) {
				printMessage("");
				printMessage("An error occurred!");
				logException(e);
			}

		}

		printMessage("");
		printMessage("Terminal \'" + terminalName + "\' finished after " + (transactionCount - 1) + " transaction(s).");

		parent.signalTerminalEnded(this, newOrderCounter);
	}

	public void stopRunningWhenPossible() {
		stopRunningSignal = true;
		printMessage("");
		printMessage("Terminal received stop signal!");
		printMessage("Finishing current transaction before exit...");
	}

	private void executeTransactions(int numTransactions) {
		boolean stopRunning = false;
		int k = 0;
		int l = 0;

		if (numTransactions != -1)
			printMessage("Executing " + numTransactions + " transactions...");
		else
			printMessage("Executing for a limited time...");

		for (int i = 0; (i < numTransactions || numTransactions == -1) && !stopRunning; i++) {
			long transactionType = rnd.nextLong(1, 100);
			int skippedDeliveries = 0, newOrder = 0;
			String transactionTypeName = "New-Order";
			long transactionStart = System.currentTimeMillis();

			/*
			 * TPC/C specifies that each terminal has a fixed
			 * "home" warehouse. However, since this implementation
			 * does not simulate "terminals", but rather simulates
			 * "application threads", that association is no longer
			 * valid. In the case of having less clients than
			 * warehouses (which should be the normal case), it
			 * leaves the warehouses without a client without any
			 * significant traffic, changing the overall database
			 * access pattern significantly.
			 */
			if (!terminalWarehouseFixed)
				terminalWarehouseID = rnd.nextInt(1, numWarehouses);

			int ooid = 0;

			if (!finishNewOrder) {
				if (reservedOOid != -1 && retryCount < 10) {
					ooid = reservedOOid;
					parent.addRetryNum();
					if (retryCount == 0) {
						parent.addAbortNum();
					}
					retryCount++;
				} else {
					synchronized (lock) {
						ooid = jTPCC.ooid++;
					}
					retryCount = 0;
				}
				if (ooid >= csv.numo || ooid < 0) {
					finishNewOrder = true;
				}
			}

			// 执行指定New Order事务
			if (!finishNewOrder) {
				jTPCCTData term = new jTPCCTData();
				term.setTerminalId(terminalId);
				try {
					term.generateNewOrder(log, rnd, 0, ooid);
					term.traceScreen(log);
					term.execute(log, db, this.parent);
					parent.resultAppend(term);
					term.traceScreen(log);

					if (concurrencyControl.equals("aria") || parent.shouldRetry()) {
						reservedOOid = term.getReservedOOID();
					}

				} catch (Exception e) {
					log.fatal(e.getMessage());
					e.printStackTrace();
					System.exit(4);
				}

				transactionTypeName = "New-Order";

				if (reservedOOid == -1) {
					newOrderCounter++;
					synchronized (lock) {
						// jTPCC.olstr += csv.oorder.oo_ol_cnt.get(ooid);
						jTPCC.olstr++;
					}
				}
				newOrder = 1;
			} else if (transactionType <= newOrderWeight) {
				jTPCCTData term = new jTPCCTData();
				term.setTerminalId(terminalId);
				term.setNumWarehouses(numWarehouses);
				term.setWarehouse(terminalWarehouseID);
				term.setDistrict(terminalDistrictID);
				try {
					term.generateNewOrder(log, rnd, 0);
					term.traceScreen(log);
					term.execute(log, db, this.parent);
					parent.resultAppend(term);
					term.traceScreen(log);
				} catch (Exception e) {
					log.fatal(e.getMessage());
					e.printStackTrace();
					System.exit(4);
				}
				transactionTypeName = "New-Order";
				newOrderCounter++;
				newOrder = 1;
			} else if (transactionType <= newOrderWeight + paymentWeight) {
				jTPCCTData term = new jTPCCTData();
				term.setTerminalId(terminalId);
				term.setNumWarehouses(numWarehouses);
				term.setWarehouse(terminalWarehouseID);
				term.setDistrict(terminalDistrictID);
				try {
					term.generatePayment(log, rnd, 0);
					term.traceScreen(log);
					term.execute(log, db, this.parent);
					parent.resultAppend(term);
					term.traceScreen(log);
				} catch (Exception e) {
					log.fatal(e.getMessage());
					e.printStackTrace();
					System.exit(4);
				}
				transactionTypeName = "Payment";
			} else if (transactionType <= newOrderWeight + paymentWeight + stockLevelWeight) {
				jTPCCTData term = new jTPCCTData();
				term.setTerminalId(terminalId);
				term.setNumWarehouses(numWarehouses);
				term.setWarehouse(terminalWarehouseID);
				term.setDistrict(terminalDistrictID);
				try {
					term.generateStockLevel(log, rnd, 0);
					term.traceScreen(log);
					term.execute(log, db, this.parent);
					parent.resultAppend(term);
					term.traceScreen(log);
				} catch (Exception e) {
					log.fatal(e.getMessage());
					e.printStackTrace();
					System.exit(4);
				}
				transactionTypeName = "Stock-Level";
			} else if (transactionType <= newOrderWeight + paymentWeight + stockLevelWeight + orderStatusWeight) {
				jTPCCTData term = new jTPCCTData();
				term.setTerminalId(terminalId);
				term.setNumWarehouses(numWarehouses);
				term.setWarehouse(terminalWarehouseID);
				term.setDistrict(terminalDistrictID);
				try {
					term.generateOrderStatus(log, rnd, 0);
					term.traceScreen(log);
					term.execute(log, db, this.parent);
					parent.resultAppend(term);
					term.traceScreen(log);
				} catch (Exception e) {
					log.fatal(e.getMessage());
					e.printStackTrace();
					System.exit(4);
				}
				transactionTypeName = "Order-Status";
			} else if (transactionType <= newOrderWeight + paymentWeight + stockLevelWeight + orderStatusWeight
					+ deliveryWeight) {
				jTPCCTData term = new jTPCCTData();
				term.setTerminalId(terminalId);
				term.setNumWarehouses(numWarehouses);
				term.setWarehouse(terminalWarehouseID);
				term.setDistrict(terminalDistrictID);
				try {
					term.generateDelivery(log, rnd, 0);
					term.traceScreen(log);
					term.execute(log, db, this.parent);
					parent.resultAppend(term);
					term.traceScreen(log);

					/*
					 * The old style driver does not have a delivery
					 * background queue, so we have to execute that
					 * part here as well.
					 */
					jTPCCTData bg = term.getDeliveryBG();
					bg.traceScreen(log);
					bg.execute(log, db, this.parent);
					parent.resultAppend(bg);
					bg.traceScreen(log);

					skippedDeliveries = bg.getSkippedDeliveries();
				} catch (Exception e) {
					log.fatal(e.getMessage());
					e.printStackTrace();
					System.exit(4);
				}
				transactionTypeName = "Delivery";
			} else if (false) {
				jTPCCTData term = new jTPCCTData();
				term.setNumWarehouses(numWarehouses);
				term.setWarehouse(terminalWarehouseID);
				// term.setDistrict(terminalDistrictID);
				try {
					if (same[i] < 20 || k >= numTransactions * 0.8 && l < numTransactions * 0.2) {
						term.generateNewOrder(log, rnd, 0);
						l++;
					} else {
						term.generateNewOrder(log, rnd1[k], 0);
						k++;
					}
					term.traceScreen(log);
					term.execute(log, db, this.parent);
					parent.resultAppend(term);
					term.traceScreen(log);
				} catch (Exception e) {
					log.fatal(e.getMessage());
					e.printStackTrace();
					System.exit(4);
				}
				transactionTypeName = "New-Order";
				newOrderCounter++;
				newOrder = 1;
			}

			long transactionEnd = System.currentTimeMillis();

			if (!transactionTypeName.equals("Delivery")) {
				parent.signalTerminalEndedTransaction(this.terminalName, transactionTypeName, transactionEnd - transactionStart,
						null, newOrder);
			} else {
				parent.signalTerminalEndedTransaction(this.terminalName, transactionTypeName, transactionEnd - transactionStart,
						(skippedDeliveries == 0 ? "None" : "" + skippedDeliveries + " delivery(ies) skipped."), newOrder);
			}

			if (limPerMin_Terminal > 0) {
				long elapse = transactionEnd - transactionStart;
				long timePerTx = 60000 / limPerMin_Terminal;

				if (elapse < timePerTx) {
					try {
						long sleepTime = timePerTx - elapse;
						Thread.sleep((sleepTime));
					} catch (Exception e) {
					}
				}
			}
			if (stopRunningSignal)
				stopRunning = true;
		} // END 'for' LOOP

	}

	private void error(String type) {
		log.error(terminalName + ", TERMINAL=" + terminalName + "  TYPE=" + type + "  COUNT=" + transactionCount);
		System.out.println(terminalName + ", TERMINAL=" + terminalName + "  TYPE=" + type + "  COUNT=" + transactionCount);
	}

	private void logException(Exception e) {
		StringWriter stringWriter = new StringWriter();
		PrintWriter printWriter = new PrintWriter(stringWriter);
		e.printStackTrace(printWriter);
		printWriter.close();
		log.error(stringWriter.toString());
	}

	private void terminalMessage(String message) {
		log.trace(terminalName + ", " + message);
	}

	private void printMessage(String message) {
		log.trace(terminalName + ", " + message);
	}

	void transRollback() {
		try {
			conn.rollback();
		} catch (SQLException se) {
			log.error(se.getMessage());
		}
	}

	void transCommit() {
		try {
			conn.commit();
		} catch (SQLException se) {
			log.error(se.getMessage());
			transRollback();
		}
	} // end transCommit()

}
