/*
 * jTPCC - Open Source Java implementation of a TPC-C like benchmark
 *
 * Copyright (C) 2003, Raul Barbosa
 * Copyright (C) 2004-2016, Denis Lussier
 * Copyright (C) 2016, Jan Wieck
 *
 */

import org.apache.log4j.*;
import org.firebirdsql.jdbc.parser.JaybirdSqlParser.nullValue_return;

import java.io.*;
import java.nio.file.*;
import java.sql.*;
import java.util.*;
import java.text.*;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.ThreadLocalRandom;

import java.time.LocalDateTime;

public class jTPCC implements jTPCCConfig {
    private static org.apache.log4j.Logger log = org.apache.log4j.Logger.getLogger(jTPCC.class);
    private static String resultDirName = null;
    private static BufferedWriter resultCSV = null;
    private static BufferedWriter runInfoCSV = null;
    private static int runID = 0;

    private static int numTerminals = -1; // #RAIN

    private static final int PREDICTION_CACHE_SIZE = 2000; // #RAIN
    private static Integer k_value = 2; // #RAIN
    private static AtomicInteger waitNumber = new AtomicInteger(0); // #RAIN
    private static AtomicInteger retryNumber = new AtomicInteger(0); // #RAIN
    private static AtomicInteger abortNumber = new AtomicInteger(0); // #RAIN
    private static AtomicInteger commitNumber = new AtomicInteger(0); // #RAIN
    private static AtomicInteger timeoutNumber = new AtomicInteger(0); // #RAIN
    private static AtomicInteger errorNumber = new AtomicInteger(0); // #RAIN

    private int dbType = DB_UNKNOWN;
    private int currentlyDisplayedTerminal;

    private int[] i_ids; // #RAIN
    private jTPCCTransaction[] runningPool; // #RAIN
    private List<jTPCCLog> transactionLog; // #RAIN
    private List<jTPCCLog> currentLog; // #RAIN

    private jTPCCTransaction[] predictionCache; // #RAIN
    private jTPCCTransaction[] predictionCache2; // #RAIN
    private jTPCCTransaction[] predictionCache3; // #RAIN

    private ReentrantLock[] locks; // #RAIN
    private Condition[] conditions; // #RAIN

    private long predictTime = (long) 0; // #RAIN
    private long lockTime = (long) 0; // #RAIN
    private long execTime = (long) 0; // #RAIN
    private long abortTime = (long) 0; // #RAIN

    private jTPCCTerminal[] terminals;
    private String[] terminalNames;
    private boolean terminalsBlockingExit = false;
    private long terminalsStarted = 0, sessionCount = 0, transactionCount = 0;
    private Object counterLock = new Object();

    private long newOrderCounter = 0, sessionStartTimestamp, sessionEndTimestamp, sessionNextTimestamp = 0,
            sessionNextKounter = 0;
    private long sessionEndTargetTime = -1, fastNewOrderCounter, recentTpmC = 0, recentTpmTotal = 0;
    private boolean signalTerminalsRequestEndSent = false, databaseDriverLoaded = false;

    private FileOutputStream fileOutputStream;
    private PrintStream printStreamReport;
    private String sessionStart, sessionEnd;
    private int limPerMin_Terminal;

    private double tpmC;
    private jTPCCRandom rnd;
    private OSCollector osCollector = null;

    public static String runfilepath = null;

    public static String concurrencyControl = null; // #RAIN
    public static boolean doRetry = false; // #RAIN
    public static boolean doConflictDetection = false; // #RAIN
    public static boolean doLogAnalysis = false; // #RAIN
    public static boolean doHashInMogi = false; // #RAIN

    public static jTPCCDataTS dataTimestamps = null; // #RAIN
    public static Long currentTS = (long) 0; // #RAIN

    public static ConcurrentHashMap<String, List<Long>> latencyRecords = new ConcurrentHashMap<>(); // #RAIN

    public static Loadcsv csv;
    public static int ooid = 0;
    public static int olstr = 0;
    public static int runtime = 0;

    public static void main(String args[]) {
        PropertyConfigurator.configure("log4j.properties");
        new jTPCC();
    }

    private String getProp(Properties p, String pName) {
        String prop = p.getProperty(pName);
        log.info("Term-00, " + pName + "=" + prop);
        return (prop);
    }

    /* --------------------------------- #RAIN --------------------------------- */

    public static void exportLatencyStatsToCSV(String filename) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename), true)) {
            writer.println("txn_type,count,avg_ms,p95_ms,p99_ms,max_ms");

            for (Map.Entry<String, List<Long>> entry : latencyRecords.entrySet()) {
                String txn = entry.getKey();
                List<Long> data = entry.getValue();
                if (data.isEmpty())
                    continue;

                Collections.sort(data);
                int size = data.size();
                long avg = data.stream().mapToLong(Long::longValue).sum() / size;
                long p95 = data.get((int) (size * 0.95));
                long p99 = data.get((int) (size * 0.99));
                long max = data.get(size - 1);

                writer.printf("%s,%d,%d,%d,%d,%d%n", txn, size, avg, p95, p99, max);
            }
        }
    }

    public void addToAbortTime(long addedAbortTime) {
        abortTime += addedAbortTime;
    }

    public void addToLockTime(long addedLockTime) {
        lockTime += addedLockTime;
    }

    public void addToExecTime(long addedExecTime) {
        execTime += addedExecTime;
    }

    public boolean needsLogAnalysis() {
        return doLogAnalysis;
    }

    public boolean needsConflictDetection() {
        return doConflictDetection;
    }

    public boolean shouldRetry() {
        return doRetry;
    }

    public boolean needsHashForMogi() {
        return doHashInMogi;
    }

    public void addTransactionLog(int ooid, jTPCCDataitem dataitem) {
        jTPCCLog log = new jTPCCLog(ooid, dataitem);

        synchronized (transactionLog) {
            transactionLog.add(log);
        }
    }

    public void removeTransactionLog(int ooid) {
        synchronized (transactionLog) {
            transactionLog.removeIf(item -> (item.getOoid() == ooid));
        }
    }

    public void addCurrentLog(int ooid, jTPCCDataitem dataitem) {
        jTPCCLog log = new jTPCCLog(ooid, dataitem);
        synchronized (currentLog) {
            currentLog.add(log);
        }
    }

    public void removeCurrentLog(int ooid, boolean onlyRead) {
        synchronized (currentLog) {
            if (!onlyRead)
                currentLog.removeIf(item -> (item.getOoid() == ooid));
            else
                currentLog.removeIf(item -> (item.getOoid() == ooid && item.isRead()));
        }
    }

    public boolean findCurrentLog(jTPCCDataitem dataitem) {
        synchronized (currentLog) {
            for (jTPCCLog log : currentLog) {
                if (log.getDataitem().equals(dataitem) && (log.getDataitem().getTypeId() == dataitem.getTypeId())) {
                    return true;
                }
            }
        }

        return false;
    }

    public jTPCCDataTS getDataTimestamps() {
        return dataTimestamps;
    }

    public void setKValue(int k) {
        k_value = k;
    }

    public static List<jTPCCDataitem> extractDataitems(String input) {
        Pattern pattern = Pattern.compile("\\((\\d+),\\s*(\\d+),\\s*(\\d+),\\s*(\\d+),\\s*(\\d+),\\s*(\\d+)\\)");
        Matcher matcher = pattern.matcher(input);

        List<jTPCCDataitem> items = new ArrayList<>();

        while (matcher.find()) {
            jTPCCDataitem item = new jTPCCDataitem();
            item.setTypeId(Integer.parseInt(matcher.group(1)));
            item.setTableId(Integer.parseInt(matcher.group(2)));
            item.setWid(Integer.parseInt(matcher.group(3)));
            item.setDid(Integer.parseInt(matcher.group(4)));
            item.setCid(Integer.parseInt(matcher.group(5)));
            item.setIid(Integer.parseInt(matcher.group(6)));
            items.add(item);
        }

        return items;
    }

    public jTPCCTransaction[] readCacheCSV(String csvFile) {
        String line;

        jTPCCTransaction[] cacheTable = new jTPCCTransaction[PREDICTION_CACHE_SIZE];

        for (int k = 0; k < cacheTable.length; k++) {
            cacheTable[k] = new jTPCCTransaction(-1);
        }

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            // Read the header line
            br.readLine();

            while ((line = br.readLine()) != null) {
                // Use comma as separator
                String[] columns = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");

                String inputString = columns[0].replaceAll("\"", "").trim();
                String outputString = columns[1].replaceAll("\"", "").trim();
                int count = Integer.parseInt(columns[2].trim());

                long inputHashValue = 0;
                long outputHashValue = 0;

                List<jTPCCDataitem> inputVectors = extractDataitems(inputString);
                List<jTPCCDataitem> outputVectors = extractDataitems(outputString);

                jTPCCTransaction transaction = new jTPCCTransaction();

                transaction.setOoid(0);
                transaction.setInputSize(inputVectors.size());
                transaction.setOutputSize(outputVectors.size());

                for (int i = 0; i < inputVectors.size(); i++) {
                    inputHashValue += inputVectors.get(i).getHashValue();
                }

                for (int j = 0; j < outputVectors.size(); j++) {
                    outputHashValue += outputVectors.get(j).getHashValue();
                }

                transaction.setInputHash(inputHashValue);
                transaction.setOutputHash(outputHashValue);

                List<jTPCCDataitem> combinedList = new ArrayList<>(inputVectors);
                combinedList.addAll(outputVectors);
                transaction.setList(combinedList);

                int index = (int) (inputHashValue % PREDICTION_CACHE_SIZE);

                while (cacheTable[index].getOoid() != -1) {
                    index++;

                    if (index == PREDICTION_CACHE_SIZE)
                        index = 0;

                    if (index == (int) (inputHashValue % PREDICTION_CACHE_SIZE))
                        break;
                }

                cacheTable[index] = transaction;
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        return cacheTable;
    }

    public List<Integer> checkIntersection(int terminalId) {
        if (terminalId > runningPool.length || terminalId < 0) {
            throw new IllegalArgumentException("terminalId out of the range.");
        }

        if (runningPool[terminalId - 1] == null) {
            throw new NullPointerException("runningPool[terminalId-1] is null");
        }

        List<jTPCCDataitem> temp_list = new ArrayList<>(runningPool[terminalId - 1].getList());
        int current_inputSize = runningPool[terminalId - 1].getInputSize();

        if (temp_list == null) {
            throw new NullPointerException("temp_list is null");
        }

        List<Integer> indicesWithTrueIntersections = new ArrayList<>();

        // 分离前 current_inputSize 项 (Case 1) 和剩余其他项 (Case 2)
        List<jTPCCDataitem> case1List = temp_list.subList(0, current_inputSize);
        List<jTPCCDataitem> case2List = temp_list.subList(current_inputSize, temp_list.size());

        if (needsHashForMogi()) {
            // Hash
            Map<jTPCCDataitem, Integer> terminalMapCase1 = new HashMap<>();
            Map<jTPCCDataitem, Integer> terminalMapCase2 = new HashMap<>();

            for (jTPCCDataitem item : case1List) {
                if (item == null)
                    continue;
                terminalMapCase1.put(item, item.getTypeId());
            }

            for (jTPCCDataitem item : case2List) {
                if (item == null)
                    continue;
                terminalMapCase2.put(item, item.getTypeId());
            }

            for (int i = 0; i < runningPool.length; i++) {
                if (i == terminalId - 1 || runningPool[i].getInputSize() < 1) {
                    continue;
                }

                List<jTPCCDataitem> currentList;

                synchronized (runningPool[i]) {
                    if (runningPool[i].getOoid() <= 0)
                        continue;
                    currentList = new ArrayList<>(runningPool[i].getList());
                }

                boolean hasTrueIntersection = false;
                boolean hasCase1Intersection = false;
                boolean hasCase2Intersection = false;

                for (jTPCCDataitem currentItem : currentList) {
                    Integer terminalTypeIdCase1 = terminalMapCase1.get(currentItem);
                    if (terminalTypeIdCase1 != null && (terminalTypeIdCase1 + currentItem.getTypeId() > 2)) {
                        hasCase1Intersection = true;
                        break;
                    }
                }

                if (hasCase1Intersection) {
                    // Case 1: 前 current_inputSize 项有交集
                    indicesWithTrueIntersections.add(i);
                    continue; // 如果已经有 Case 1 的交集，则跳过 Case 2 的检查
                }

                for (jTPCCDataitem currentItem : currentList) {
                    Integer terminalTypeIdCase2 = terminalMapCase2.get(currentItem);
                    if (terminalTypeIdCase2 != null && (terminalTypeIdCase2 + currentItem.getTypeId() > 2)) {
                        hasCase2Intersection = true;
                        break;
                    }
                }

                if (hasCase2Intersection) {
                    // Case 2: 剩余其他项有交集
                    indicesWithTrueIntersections.add(i);
                }
            }
        } else {
            // Ergodic
            for (int i = 0; i < runningPool.length; i++) {
                if (i == terminalId - 1 || runningPool[i].getInputSize() < 1) {
                    continue;
                }

                List<jTPCCDataitem> currentList;

                synchronized (runningPool[i]) {
                    if (runningPool[i].getOoid() <= 0)
                        continue;
                    currentList = new ArrayList<>(runningPool[i].getList());
                }

                boolean hasTrueIntersection = false;
                boolean hasCase1Intersection = false;
                boolean hasCase2Intersection = false;

                for (jTPCCDataitem currentItem : currentList) {
                    for (jTPCCDataitem terminalItem : case1List) {
                        if (terminalItem != null && currentItem != null
                                && terminalItem.equals(currentItem)
                                && (terminalItem.getTypeId() + currentItem.getTypeId() > 2)) {
                            hasCase1Intersection = true;
                            break;
                        }
                    }

                    if (hasCase1Intersection) {
                        break;
                    }
                }

                if (hasCase1Intersection) {
                    // Case 1: 前 current_inputSize 项有交集
                    indicesWithTrueIntersections.add(i);
                    continue; // 如果已经有 Case 1 的交集，则跳过 Case 2 的检查
                }

                for (jTPCCDataitem currentItem : currentList) {
                    for (jTPCCDataitem terminalItem : case2List) {
                        if (terminalItem != null && currentItem != null
                                && terminalItem.equals(currentItem)
                                && (terminalItem.getTypeId() + currentItem.getTypeId() > 2)) {
                            hasCase2Intersection = true;
                            break;
                        }
                    }

                    if (hasCase2Intersection) {
                        break;
                    }
                }

                if (hasCase2Intersection) {
                    // Case 2: 剩余其他项有交集
                    indicesWithTrueIntersections.add(i);
                }
            }
        }

        return indicesWithTrueIntersections;
    }

    public boolean checkIntersection(int terminalId, List<Integer> trueConflictedTxns) {
        if (terminalId > runningPool.length || terminalId < 0) {
            throw new IllegalArgumentException("terminalId out of the range.");
        }

        jTPCCTransaction terminalTransaction = runningPool[terminalId - 1];
        if (terminalTransaction == null) {
            throw new NullPointerException("runningPool[terminalId-1] is null");
        }

        List<jTPCCDataitem> tempList;

        synchronized (terminalTransaction) {
            tempList = new ArrayList<>(terminalTransaction.getList());
        }

        int currentInputSize = terminalTransaction.getInputSize();

        if (tempList == null) {
            throw new NullPointerException("tempList is null");
        }

        boolean needAborting = false;

        if (needsHashForMogi()) {
            Map<jTPCCDataitem, Integer> terminalMap = new HashMap<>(tempList.size());

            for (jTPCCDataitem item : tempList) {
                if (item != null) {
                    terminalMap.put(item, item.getTypeId());
                }
            }

            for (int i = 0; i < runningPool.length; i++) {
                if (i == terminalId - 1) {
                    continue; // Skip the terminal itself
                }

                jTPCCTransaction currentTransaction = runningPool[i];
                if (currentTransaction == null || currentTransaction.getInputSize() < 1
                        || currentTransaction.getOoid() <= 0) {
                    continue; // Skip invalid or empty transactions
                }

                List<jTPCCDataitem> currentList = currentTransaction.getList();

                // Compare the first part (Case 1)
                for (int j = 0; j < currentInputSize && !needAborting; j++) {
                    jTPCCDataitem currentItem = tempList.get(j);
                    Integer terminalTypeId = terminalMap.get(currentItem);
                    if (terminalTypeId != null && (terminalTypeId + currentItem.getTypeId() > 2)) {
                        return true;
                    }
                }

                // Compare the remaining part (Case 2)
                for (int j = currentInputSize; j < tempList.size(); j++) {
                    jTPCCDataitem currentItem = tempList.get(j);
                    Integer terminalTypeId = terminalMap.get(currentItem);
                    if (terminalTypeId != null && (terminalTypeId + currentItem.getTypeId() > 2)) {
                        trueConflictedTxns.add(i);
                        break;
                    }
                }
            }

            return needAborting;
        } else {
            // Ergodic (traversal) method
            for (int i = 0; i < runningPool.length; i++) {
                if (i == terminalId - 1 || runningPool[i].getInputSize() < 1) {
                    continue;
                }

                List<jTPCCDataitem> currentList;

                synchronized (runningPool[i]) {
                    if (runningPool[i].getOoid() <= 0) {
                        continue;
                    }
                    currentList = new ArrayList<>(runningPool[i].getList());
                }

                boolean needWaiting = false;

                // Compare the first part (Case 1)
                for (int j = 0; j < currentInputSize && !needAborting; j++) {
                    jTPCCDataitem terminalItem = tempList.get(j);
                    for (jTPCCDataitem currentItem : currentList) {
                        if (terminalItem != null && currentItem != null
                                && terminalItem.equals(currentItem)
                                && (terminalItem.getTypeId() + currentItem.getTypeId() > 2)) {
                            return true;
                        }
                    }
                }

                // Compare the remaining part (Case 2)
                for (int j = currentInputSize; j < tempList.size() && !needWaiting; j++) {
                    jTPCCDataitem terminalItem = tempList.get(j);
                    for (jTPCCDataitem currentItem : currentList) {
                        if (terminalItem != null && currentItem != null
                                && terminalItem.equals(currentItem)
                                && (terminalItem.getTypeId() + currentItem.getTypeId() > 2)) {
                            needWaiting = true;
                            trueConflictedTxns.add(i);
                            break;
                        }
                    }
                }
            }
        }

        return needAborting;
    }

    public List<jTPCCDataitem> predict(List<jTPCCDataitem> list, int inputSize) {
        int iSize = inputSize;

        if (list.size() < iSize) {
            iSize = list.size();
        }

        long hashValue = 0;

        for (int i = 0; i < iSize; i++) {
            hashValue += list.get(i).getHashValue();
        }

        int index = (int) (hashValue % PREDICTION_CACHE_SIZE);

        while (predictionCache[index].getOoid() != -1) {
            if (predictionCache[index].getInputHash() == hashValue) {
                long endPreTime = System.currentTimeMillis();
                return predictionCache[index].getList();
            }

            index++;

            if (index == PREDICTION_CACHE_SIZE)
                index = 0;

            if (index == (int) (hashValue % PREDICTION_CACHE_SIZE))
                break;
        }

        return list;
    }

    public void addTransactionToRunningPool(int terminalId, jTPCCTransaction transaction) {
        runningPool[terminalId - 1] = transaction;
    }

    public boolean addDataitemToTransaction(int terminalId, jTPCCDataitem data) {
        boolean isTimeout = false;

        if (runningPool[terminalId - 1] == null)
            return isTimeout;

        int inputSize = runningPool[terminalId - 1].getInputSize();
        int outputSize = runningPool[terminalId - 1].getOutputSize();

        runningPool[terminalId - 1].addDataToList(data, true);
        runningPool[terminalId - 1].setInputSize(inputSize + 1);

        if (inputSize + 1 == k_value) {
            // List<Integer> conflictedTxns = new ArrayList<>();

            long startPreTime = System.currentTimeMillis();

            runningPool[terminalId - 1].setList(predict(runningPool[terminalId - 1].getList(), inputSize + 1));

            // boolean needAborting = checkIntersection(terminalId, conflictedTxns);

            List<Integer> conflictedTxns = checkIntersection(terminalId);

            long endPreTime = System.currentTimeMillis();

            this.predictTime += (endPreTime - startPreTime + 1);

            boolean needsWaiting = false;

            for (Integer tId : conflictedTxns) {
                if (tId < 0 || tId >= numTerminals)
                    continue;

                if (terminals[tId] == null)
                    continue;

                if (terminals[tId].checkWaiting())
                    continue;

                Integer waitedTID = terminalId - 1;
                terminals[tId].addToWaitedTxns(waitedTID);
                needsWaiting = true;
            }

            if (needsWaiting) {
                locks[terminalId - 1].lock();

                try {
                    addWaitNum();

                    terminals[terminalId - 1].setWaiting(true);

                    long lockStart = System.currentTimeMillis();

                    conditions[terminalId - 1] = locks[terminalId - 1].newCondition();

                    if (!conditions[terminalId - 1].await(50, TimeUnit.MILLISECONDS)) {
                        addTimeoutNum();
                        isTimeout = true;
                    }

                    long lockEnd = System.currentTimeMillis();
                    this.addToLockTime(lockEnd - lockStart + 1);

                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    terminals[terminalId - 1].setWaiting(false);
                    locks[terminalId - 1].unlock();
                }
            }

        } else if (inputSize + 1 > k_value) {
            runningPool[terminalId - 1].setOutputSize(outputSize - 1);
        }

        return isTimeout;
    }

    public void commitTxnInMogi(int terminalId) {
        List<Integer> waitedTxns;

        waitedTxns = new ArrayList<>(terminals[terminalId - 1].getWaitedTxns());

        addTransactionToRunningPool(terminalId, new jTPCCTransaction());

        for (Integer tId : waitedTxns) {
            if (tId >= 0 && tId < numTerminals) {
                locks[tId].lock();
                conditions[tId].signalAll();
                locks[tId].unlock();
            }
        }

        terminals[terminalId - 1].clearWaitedTxns();
    }

    public void logConflictAnalyze() {
        List<jTPCCLog> scan = new ArrayList<>();

        for (jTPCCLog log : transactionLog) {
            if (log.isRead()) {
                for (jTPCCLog temp_log : scan) {
                    if (temp_log.getDataitem().equals(log.getDataitem())) {
                        addErrorNum();
                        break;
                    }
                }

                scan.add(log);
            } else if (log.isWrite()) {
                for (jTPCCLog temp_log : scan) {
                    if (temp_log.getOoid() == log.getOoid()) {
                        scan.remove(temp_log);
                        break;
                    }
                }
            } else {

            }
        }
    }

    public void addRetryNum() {
        retryNumber.incrementAndGet();
    }

    public int getRetryNum() {
        return retryNumber.get();
    }

    public void addAbortNum() {

        abortNumber.incrementAndGet();
    }

    public int getAbortNum() {
        return abortNumber.get();
    }

    public void addCommitNum() {
        commitNumber.incrementAndGet();
    }

    public int getCommitNum() {
        return commitNumber.get();
    }

    public void addWaitNum() {
        waitNumber.incrementAndGet();
    }

    public int getWaitNum() {
        return waitNumber.get();
    }

    public void addTimeoutNum() {
        timeoutNumber.incrementAndGet();
    }

    public int getTimeoutNum() {
        return timeoutNumber.get();
    }

    public void addErrorNum() {
        errorNumber.incrementAndGet();
    }

    public int getErrorNum() {
        return errorNumber.get();
    }

    public boolean terminalsClose() {
        if (this.ooid + 1 >= csv.numo)
            return true;
        else
            return false;
    }

    public int get_i_id(int ooid) {
        synchronized (i_ids) {
            if (i_ids.length > ooid)
                return i_ids[ooid];
            else
                return -1;
        }
    }

    public void lockIID(int ooid) {
        if (get_i_id(ooid) != -1)
            locks[get_i_id(ooid)].lock();
    }

    public void unlockIID(int ooid) {
        // System.out.println("UNLOCK: " + get_i_id(ooid));

        if (get_i_id(ooid) != -1) {
            if (locks[get_i_id(ooid)].isHeldByCurrentThread()) {
                locks[get_i_id(ooid)].unlock();
            }
        }

    }

    public void lockOOID(int ooid) {
        if (ooid != -1)
            locks[ooid].lock();
    }

    public void unlockOOID(int ooid) {
        if (ooid != -1)
            locks[ooid].unlock();
    }

    // public void awaitConditionbyOOID(int ooid) throws InterruptedException
    // {
    // if(ooid != -1)
    // conditions[ooid].await(50, TimeUnit.MILLISECONDS);
    // }

    // public void signalConditionbyOOID(int ooid)
    // {
    // if(ooid != -1)
    // conditions[ooid].signalAll();
    // }

    /* --------------------------------- #RAIN --------------------------------- */

    public jTPCC() {
        // csv = new Loadcsv();
        // load the ini file
        Properties ini = new Properties();
        try {
            ini.load(new FileInputStream(System.getProperty("prop")));
        } catch (IOException e) {
            errorMessage("Term-00, could not load properties file");
        }

        log.info("Term-00, ");
        log.info("Term-00, +-------------------------------------------------------------+");
        log.info("Term-00,      BenchmarkSQL v" + JTPCCVERSION);
        log.info("Term-00, +-------------------------------------------------------------+");
        log.info("Term-00,  (c) 2003, Raul Barbosa");
        log.info("Term-00,  (c) 2004-2016, Denis Lussier");
        log.info("Term-00,  (c) 2016, Jan Wieck");
        log.info("Term-00, +-------------------------------------------------------------+");
        log.info("Term-00, ");
        String iDB = getProp(ini, "db");
        String iDriver = getProp(ini, "driver");
        String iConn = getProp(ini, "conn");
        String iUser = getProp(ini, "user");
        String iKValue = getProp(ini, "kValue").trim(); // #RAIN
        String iConcurrencyControl = getProp(ini, "concurrencyControl").trim(); // #RAIN
        String iNeedsRetry = getProp(ini, "needsRetry").trim(); // #RAIN
        String iNeedsDetection = getProp(ini, "conflictDetection").trim(); // #RAIN
        String iLogAnalysis = getProp(ini, "logAnalysis").trim(); // #RAIN
        String iUseHash = getProp(ini, "useHashForMogi").trim(); // #RAIN
        String iPassword = ini.getProperty("password");
        runfilepath = ini.getProperty("runfilepath");
        runtime = Integer.parseInt(ini.getProperty("runtime"));
        concurrencyControl = iConcurrencyControl;

        if ((iNeedsRetry.equals("on") || iNeedsRetry.equals("true"))) {
            doRetry = true;
        }

        if ((iNeedsDetection.equals("on") || iNeedsDetection.equals("true"))) {
            doConflictDetection = true;
        }

        if ((iLogAnalysis.equals("on") || iLogAnalysis.equals("true"))) {
            doLogAnalysis = true;
        }

        if ((iUseHash.equals("on") || iUseHash.equals("true"))) {
            doHashInMogi = true;
        }

        transactionLog = new ArrayList<>(); // #RAIN
        currentLog = new ArrayList<>(); // #RAIN
        dataTimestamps = new jTPCCDataTS(); // #RAIN

        lockTime = 0; // #RAIN
        execTime = 0; // #RAIN

        latencyRecords.put("NEW_ORDER", Collections.synchronizedList(new ArrayList<>())); // #RAIN
        latencyRecords.put("PAYMENT", Collections.synchronizedList(new ArrayList<>())); // #RAIN
        latencyRecords.put("DELIVERY", Collections.synchronizedList(new ArrayList<>())); // #RAIN

        csv = new Loadcsv();

        log.info("Term-00, ");
        String runPath = getProp(ini, "runfilepath");
        String iWarehouses = getProp(ini, "warehouses");
        String iTerminals = getProp(ini, "terminals");

        ExecutorService executor = Executors.newFixedThreadPool(Integer.parseInt(iTerminals)); // #RAIN

        String iRunTxnsPerTerminal = ini.getProperty("runTxnsPerTerminal");
        String iRunMins = ini.getProperty("runMins");
        if (Integer.parseInt(iRunTxnsPerTerminal) == 0 && Integer.parseInt(iRunMins) != 0) {
            log.info("Term-00, runMins" + "=" + iRunMins);
        } else if (Integer.parseInt(iRunTxnsPerTerminal) != 0 && Integer.parseInt(iRunMins) == 0) {
            log.info("Term-00, runTxnsPerTerminal" + "=" + iRunTxnsPerTerminal);
        } else {
            errorMessage("Term-00, Must indicate either transactions per terminal or number of run minutes!");
        }
        ;
        String limPerMin = getProp(ini, "limitTxnsPerMin");
        String iTermWhseFixed = getProp(ini, "terminalWarehouseFixed");
        log.info("Term-00, ");
        String iNewOrderWeight = getProp(ini, "newOrderWeight");
        String iPaymentWeight = getProp(ini, "paymentWeight");
        String iOrderStatusWeight = getProp(ini, "orderStatusWeight");
        String iDeliveryWeight = getProp(ini, "deliveryWeight");
        String iStockLevelWeight = getProp(ini, "stockLevelWeight");

        log.info("Term-00, ");
        String resultDirectory = getProp(ini, "resultDirectory");
        String osCollectorScript = getProp(ini, "osCollectorScript");

        log.info("Term-00, ");

        if (iDB.equals("firebird"))
            dbType = DB_FIREBIRD;
        else if (iDB.equals("oracle"))
            dbType = DB_ORACLE;
        else if (iDB.equals("postgres"))
            dbType = DB_POSTGRES;
        else {
            log.error("unknown database type '" + iDB + "'");
            return;
        }

        if (Integer.parseInt(limPerMin) != 0) {
            limPerMin_Terminal = Integer.parseInt(limPerMin) / Integer.parseInt(iTerminals);
        } else {
            limPerMin_Terminal = -1;
        }

        boolean iRunMinsBool = false;

        try {
            String driver = iDriver;
            printMessage("Loading database driver: \'" + driver + "\'...");
            Class.forName(iDriver);
            databaseDriverLoaded = true;
        } catch (Exception ex) {
            errorMessage("Unable to load the database driver!");
            databaseDriverLoaded = false;
        }

        if (databaseDriverLoaded && resultDirectory != null) {
            StringBuffer sb = new StringBuffer();
            Formatter fmt = new Formatter(sb);
            Pattern p = Pattern.compile("%t");
            Calendar cal = Calendar.getInstance();
            String iRunID;

            iRunID = System.getProperty("runID");
            if (iRunID != null) {
                runID = Integer.parseInt(iRunID);
            }

            /*
             * Split the resultDirectory into strings around
             * patterns of %t and then insert date/time formatting
             * based on the current time. That way the resultDirectory
             * in the properties file can have date/time format
             * elements like in result_%tY-%tm-%td to embed the current
             * date in the directory name.
             */
            String[] parts = p.split(resultDirectory, -1);
            sb.append(parts[0]);
            for (int i = 1; i < parts.length; i++) {
                fmt.format("%t" + parts[i].substring(0, 1), cal);
                sb.append(parts[i].substring(1));
            }
            resultDirName = sb.toString();
            File resultDir = new File(resultDirName);
            File resultDataDir = new File(resultDir, "data");

            // Create the output directory structure.
            if (!resultDir.mkdir()) {
                log.error("Failed to create directory '" +
                        resultDir.getPath() + "'");
                System.exit(1);
            }
            if (!resultDataDir.mkdir()) {
                log.error("Failed to create directory '" +
                        resultDataDir.getPath() + "'");
                System.exit(1);
            }

            // Copy the used properties file into the resultDirectory.
            try {
                Files.copy(new File(System.getProperty("prop")).toPath(),
                        new File(resultDir, "run.properties").toPath());
            } catch (IOException e) {
                log.error(e.getMessage());
                System.exit(1);
            }
            log.info("Term-00, copied " + System.getProperty("prop") +
                    " to " + new File(resultDir, "run.properties").toPath());

            // Create the runInfo.csv file.
            String runInfoCSVName = new File(resultDataDir, "runInfo.csv").getPath();
            try {
                runInfoCSV = new BufferedWriter(
                        new FileWriter(runInfoCSVName));
                runInfoCSV.write("run,driver,driverVersion,db,sessionStart," +
                        "runMins," +
                        "loadWarehouses,runWarehouses,numSUTThreads," +
                        "limitTxnsPerMin," +
                        "thinkTimeMultiplier,keyingTimeMultiplier\n");
            } catch (IOException e) {
                log.error(e.getMessage());
                System.exit(1);
            }
            log.info("Term-00, created " + runInfoCSVName + " for runID " +
                    runID);

            // Open the per transaction result.csv file.
            String resultCSVName = new File(resultDataDir, "result.csv").getPath();
            try {
                resultCSV = new BufferedWriter(new FileWriter(resultCSVName));
                resultCSV.write("run,elapsed,latency,dblatency," +
                        "ttype,rbk,dskipped,error\n");
            } catch (IOException e) {
                log.error(e.getMessage());
                System.exit(1);
            }
            log.info("Term-00, writing per transaction results to " +
                    resultCSVName);

            if (osCollectorScript != null) {
                osCollector = new OSCollector(getProp(ini, "osCollectorScript"),
                        runID,
                        Integer.parseInt(getProp(ini, "osCollectorInterval")),
                        getProp(ini, "osCollectorSSHAddr"),
                        getProp(ini, "osCollectorDevices"),
                        resultDataDir, log);
            }
            log.info("Term-00,");
        }

        if (databaseDriverLoaded) {
            try {
                boolean limitIsTime = iRunMinsBool;
                // int numTerminals = -1;
                int transactionsPerTerminal = -1;
                int numWarehouses = -1;
                int loadWarehouses = -1;
                int newOrderWeightValue = -1, paymentWeightValue = -1, orderStatusWeightValue = -1,
                        deliveryWeightValue = -1, stockLevelWeightValue = -1;
                long executionTimeMillis = -1;
                boolean terminalWarehouseFixed = true;
                long CLoad;

                Properties dbProps = new Properties();
                dbProps.setProperty("user", iUser);
                dbProps.setProperty("password", iPassword);

                /*
                 * Fine tuning of database conneciton parameters if needed.
                 */
                switch (dbType) {
                    case DB_FIREBIRD:
                        /*
                         * Firebird needs no_rec_version for our load
                         * to work. Even with that some "deadlocks"
                         * occur. Note that the message "deadlock" in
                         * Firebird can mean something completely different,
                         * namely that there was a conflicting write to
                         * a row that could not be resolved.
                         */
                        dbProps.setProperty("TRANSACTION_READ_COMMITTED",
                                "isc_tpb_read_committed," +
                                        "isc_tpb_no_rec_version," +
                                        "isc_tpb_write," +
                                        "isc_tpb_wait");
                        break;

                    default:
                        break;
                }

                try {
                    loadWarehouses = Integer.parseInt(jTPCCUtil.getConfig(iConn, dbProps, "warehouses"));
                    CLoad = Long.parseLong(jTPCCUtil.getConfig(iConn, dbProps, "nURandCLast"));
                } catch (Exception e) {
                    errorMessage(e.getMessage());
                    throw e;
                }
                this.rnd = new jTPCCRandom(CLoad);
                log.info("Term-00, C value for C_LAST during load: " + CLoad);
                log.info("Term-00, C value for C_LAST this run:    " + rnd.getNURandCLast());
                log.info("Term-00, ");

                fastNewOrderCounter = 0;
                updateStatusLine();

                try {
                    if (Integer.parseInt(iRunMins) != 0 && Integer.parseInt(iRunTxnsPerTerminal) == 0) {
                        iRunMinsBool = true;
                    } else if (Integer.parseInt(iRunMins) == 0 && Integer.parseInt(iRunTxnsPerTerminal) != 0) {
                        iRunMinsBool = false;
                    } else {
                        throw new NumberFormatException();
                    }
                } catch (NumberFormatException e1) {
                    errorMessage("Must indicate either transactions per terminal or number of run minutes!");
                    throw new Exception();
                }

                try {
                    numWarehouses = Integer.parseInt(iWarehouses);
                    if (numWarehouses <= 0)
                        throw new NumberFormatException();
                } catch (NumberFormatException e1) {
                    errorMessage("Invalid number of warehouses!");
                    throw new Exception();
                }
                if (numWarehouses > loadWarehouses) {
                    errorMessage("numWarehouses cannot be greater " +
                            "than the warehouses loaded in the database");
                    throw new Exception();
                }

                try {
                    numTerminals = Integer.parseInt(iTerminals);
                    if (numTerminals <= 0 || numTerminals > 10 * numWarehouses)
                        throw new NumberFormatException();
                } catch (NumberFormatException e1) {
                    errorMessage("Invalid number of terminals!");
                    throw new Exception();
                }

                if (Long.parseLong(iRunMins) != 0 && Integer.parseInt(iRunTxnsPerTerminal) == 0) {
                    try {
                        executionTimeMillis = Long.parseLong(iRunMins) * 60000;
                        if (executionTimeMillis <= 0)
                            throw new NumberFormatException();
                    } catch (NumberFormatException e1) {
                        errorMessage("Invalid number of minutes!");
                        throw new Exception();
                    }
                } else {
                    try {
                        transactionsPerTerminal = Integer.parseInt(iRunTxnsPerTerminal);
                        if (transactionsPerTerminal <= 0)
                            throw new NumberFormatException();
                    } catch (NumberFormatException e1) {
                        errorMessage("Invalid number of transactions per terminal!");
                        throw new Exception();
                    }
                }

                terminalWarehouseFixed = Boolean.parseBoolean(iTermWhseFixed);

                try {
                    newOrderWeightValue = Integer.parseInt(iNewOrderWeight);
                    paymentWeightValue = Integer.parseInt(iPaymentWeight);
                    orderStatusWeightValue = Integer.parseInt(iOrderStatusWeight);
                    deliveryWeightValue = Integer.parseInt(iDeliveryWeight);
                    stockLevelWeightValue = Integer.parseInt(iStockLevelWeight);

                    if (newOrderWeightValue < 0 || paymentWeightValue < 0 || orderStatusWeightValue < 0
                            || deliveryWeightValue < 0 || stockLevelWeightValue < 0)
                        throw new NumberFormatException();
                    // else if(newOrderWeightValue == 0 && paymentWeightValue == 0 &&
                    // orderStatusWeightValue == 0 && deliveryWeightValue == 0 &&
                    // stockLevelWeightValue == 0)
                    // throw new NumberFormatException();
                } catch (NumberFormatException e1) {
                    errorMessage("Invalid number in mix percentage!");
                    throw new Exception();
                }

                if (newOrderWeightValue + paymentWeightValue + orderStatusWeightValue + deliveryWeightValue
                        + stockLevelWeightValue > 100) {
                    errorMessage("Sum of mix percentage parameters exceeds 100%!");
                    throw new Exception();
                }

                newOrderCounter = 0;
                printMessage("Session started!");
                if (!limitIsTime)
                    printMessage("Creating " + numTerminals + " terminal(s) with " + transactionsPerTerminal
                            + " transaction(s) per terminal...");
                else
                    printMessage("Creating " + numTerminals + " terminal(s) with " + (executionTimeMillis / 60000)
                            + " minute(s) of execution...");
                if (terminalWarehouseFixed)
                    printMessage("Terminal Warehouse is fixed");
                else
                    printMessage("Terminal Warehouse is NOT fixed");
                printMessage("Transaction Weights: " + newOrderWeightValue + "% New-Order, " + paymentWeightValue
                        + "% Payment, " + orderStatusWeightValue + "% Order-Status, " + deliveryWeightValue
                        + "% Delivery, " + stockLevelWeightValue + "% Stock-Level");
                printMessage("Number of Terminals\t" + numTerminals);

                terminals = new jTPCCTerminal[numTerminals];
                terminalNames = new String[numTerminals];
                terminalsStarted = numTerminals;
                try {
                    String database = iConn;
                    String username = iUser;
                    String password = iPassword;

                    int[][] usedTerminals = new int[numWarehouses][10];

                    for (int i = 0; i < numWarehouses; i++) {
                        for (int j = 0; j < 10; j++) {
                            usedTerminals[i][j] = 0;
                        }
                    }

                    // transactionsPerTerminal=csv.numo/numTerminals+1;
                    for (int i = 0; i < numTerminals; i++) {
                        int terminalWarehouseID;
                        int terminalDistrictID;
                        do {
                            terminalWarehouseID = rnd.nextInt(1, numWarehouses);
                            terminalDistrictID = rnd.nextInt(1, 10);
                        } while (usedTerminals[terminalWarehouseID - 1][terminalDistrictID - 1] == 1);
                        usedTerminals[terminalWarehouseID - 1][terminalDistrictID - 1] = 1;

                        String terminalName = "Term-" + (i >= 9 ? "" + (i + 1) : "0" + (i + 1));
                        Connection conn = null;
                        printMessage("Creating database connection for " + terminalName + "...");
                        conn = DriverManager.getConnection(database, dbProps);
                        conn.setAutoCommit(false);

                        jTPCCTerminal terminal = new jTPCCTerminal(terminalName, terminalWarehouseID,
                                terminalDistrictID,
                                conn, dbType, transactionsPerTerminal, terminalWarehouseFixed, newOrderWeightValue,
                                paymentWeightValue,
                                orderStatusWeightValue,
                                deliveryWeightValue, stockLevelWeightValue, numWarehouses, limPerMin_Terminal, i, this,
                                csv, iConcurrencyControl, i + 1);

                        terminals[i] = terminal;
                        terminalNames[i] = terminalName;
                        printMessage(terminalName + "\t" + terminalWarehouseID);
                    }

                    /* ----------------------- #RAIN ----------------------- */

                    this.runningPool = new jTPCCTransaction[numTerminals];

                    for (int i = 0; i < numTerminals; i++) {
                        this.runningPool[i] = new jTPCCTransaction(-1);
                    }

                    if (iConcurrencyControl.equals("calvin") || iConcurrencyControl.equals("s2pl")) {
                        this.i_ids = new int[csv.numo];
                        this.locks = new ReentrantLock[i_ids.length + 1];

                        for (int i = 0; i < i_ids.length; i++) {
                            locks[i] = new ReentrantLock();
                        }

                        for (int i = 0; i < csv.numo; i++) {
                            i_ids[i] = csv.orderline.ol_i_id.get(i);
                        }
                    } else if (iConcurrencyControl.equals("mogi")) {
                        this.locks = new ReentrantLock[csv.numo + 1];
                        this.conditions = new Condition[csv.numo + 1];

                        for (int i = 0; i < csv.numo + 1; i++) {
                            locks[i] = new ReentrantLock();
                            conditions[i] = locks[i].newCondition();
                        }

                        this.predictionCache = readCacheCSV("/opt/data/predict_cache.csv"); // #RAIN
                    }

                    /* ----------------------- #RAIN ----------------------- */

                    sessionEndTargetTime = executionTimeMillis;
                    signalTerminalsRequestEndSent = false;

                    printMessage("Transaction\tWeight");
                    printMessage("% New-Order\t" + newOrderWeightValue);
                    printMessage("% Payment\t" + paymentWeightValue);
                    printMessage("% Order-Status\t" + orderStatusWeightValue);
                    printMessage("% Delivery\t" + deliveryWeightValue);
                    printMessage("% Stock-Level\t" + stockLevelWeightValue);
                    printMessage("Transaction Number\tTerminal\tType\tExecution Time (ms)\t\tComment");
                    printMessage("Created " + numTerminals + " terminal(s) successfully!");
                    boolean dummvar = true;

                    // Create Terminals, Start Transactions
                    sessionStart = getCurrentTime();
                    sessionStartTimestamp = System.currentTimeMillis();
                    sessionNextTimestamp = sessionStartTimestamp;
                    if (sessionEndTargetTime != -1)
                        sessionEndTargetTime += sessionStartTimestamp;

                    // Record run parameters in runInfo.csv
                    if (runInfoCSV != null) {
                        try {
                            StringBuffer infoSB = new StringBuffer();
                            Formatter infoFmt = new Formatter(infoSB);
                            infoFmt.format("%d,simple,%s,%s,%s,%s,%d,%d,%d,%d,1.0,1.0\n",
                                    runID, JTPCCVERSION, iDB,
                                    new java.sql.Timestamp(sessionStartTimestamp).toString(),
                                    iRunMins,
                                    loadWarehouses,
                                    numWarehouses,
                                    numTerminals,
                                    Integer.parseInt(limPerMin));
                            runInfoCSV.write(infoSB.toString());
                            runInfoCSV.close();
                        } catch (Exception e) {
                            log.error(e.getMessage());
                            System.exit(1);
                        }
                    }

                    printMessage("Starting all terminals...");
                    transactionCount = 1;

                    if (iConcurrencyControl.equals("aria")) {
                        while (this.ooid + 1 < csv.numo || terminalsStarted > 1) {
                            terminalsStarted = numTerminals + 1;
                            synchronized (terminals) {
                                for (int i = 0; i < terminals.length; i++) {
                                    executor.execute(terminals[i]);
                                    // System.out.println("Start terminal " + i);
                                }
                            }
                            while (terminalsStarted > 1) {
                                try {
                                    Thread.sleep(50);
                                } catch (InterruptedException e) {
                                    System.out.println(e);
                                }
                            }
                        }

                        allTerminalEnds();

                        if (!executor.awaitTermination(1, TimeUnit.SECONDS)) {
                            executor.shutdown();
                        }
                    } else if (iConcurrencyControl.equals("calvin")) {
                        while (this.ooid + 1 < csv.numo || terminalsStarted > 1) {
                            terminalsStarted = numTerminals + 1;
                            synchronized (terminals) {
                                for (int i = 0; i < terminals.length; i++) {
                                    executor.execute(terminals[i]);
                                }
                            }
                            while (terminalsStarted > 1) {
                                try {
                                    Thread.sleep(50);
                                } catch (InterruptedException e) {
                                    System.out.println(e);
                                }
                            }
                        }

                        allTerminalEnds();

                        if (!executor.awaitTermination(1, TimeUnit.SECONDS)) {
                            executor.shutdown();
                        }

                    } else if (iConcurrencyControl.equals("s2pl")) {
                        while (this.ooid + 1 < csv.numo || terminalsStarted > 1) {
                            terminalsStarted = numTerminals + 1;
                            synchronized (terminals) {
                                for (int i = 0; i < terminals.length; i++) {
                                    executor.execute(terminals[i]);
                                }
                            }
                            while (terminalsStarted > 1) {
                                try {
                                    Thread.sleep(50);
                                } catch (InterruptedException e) {
                                    System.out.println(e);
                                }
                            }
                        }

                        allTerminalEnds();

                        if (!executor.awaitTermination(1, TimeUnit.SECONDS)) {
                            executor.shutdown();
                        }
                    } else {
                        synchronized (terminals) {
                            for (int i = 0; i < terminals.length; i++) {
                                (new Thread(terminals[i])).start();
                            }
                        }
                    }

                    printMessage("All terminals started executing " + sessionStart);
                } catch (Exception e1) {
                    errorMessage("This session ended with errors!");
                    printStreamReport.close();
                    fileOutputStream.close();
                    throw new Exception();
                }
            } catch (Exception ex) {
            }
        }

        if (!iConcurrencyControl.equals("aria") && !iConcurrencyControl.equals("calvin")
                && !iConcurrencyControl.equals("s2pl")) {
            updateStatusLine();
        }
    }

    private void signalTerminalsRequestEnd(boolean timeTriggered) {
        if (!signalTerminalsRequestEndSent) {
            if (timeTriggered)
                printMessage("The time limit has been reached.");
            printMessage("Signalling all terminals to stop...");
            synchronized (terminals) {
                signalTerminalsRequestEndSent = true;

                for (int i = 0; i < terminals.length; i++) {
                    if (terminals[i] != null)
                        terminals[i].stopRunningWhenPossible();
                }
            }
            printMessage("Waiting for all active transactions to end...");
        }
    }

    public void signalTerminalEnded(jTPCCTerminal terminal, long countNewOrdersExecuted) {
        synchronized (terminals) {
            boolean found = false;
            terminalsStarted--;

            for (int i = 0; i < terminals.length && !found; i++) {
                if (terminals[i] == terminal) {
                    found = true;
                    newOrderCounter += countNewOrdersExecuted;
                    if (this.ooid + 1 >= csv.numo || (!concurrencyControl.equals("aria")
                            && !concurrencyControl.equals("calvin") && !concurrencyControl.equals("s2pl"))) {
                        terminals[i] = null;
                        terminalNames[i] = "(" + terminalNames[i] + ")";
                    }
                }
            }
        }

        if (terminalsStarted == 0) {
            sessionEnd = getCurrentTime();
            sessionEndTimestamp = System.currentTimeMillis();
            sessionEndTargetTime = -1;
            printMessage("All terminals finished executing " + sessionEnd);
            endReport();
            terminalsBlockingExit = false;
            printMessage("Session finished!");

            // If we opened a per transaction result file, close it.
            if (resultCSV != null) {
                try {
                    resultCSV.close();
                } catch (IOException e) {
                    log.error(e.getMessage());
                }
                ;
            }

            // Stop the OSCollector, if it is active.
            if (osCollector != null) {
                osCollector.stop();
                osCollector = null;
            }
        }
    }

    public void signalTerminalEndedTransaction(String terminalName, String transactionType, long executionTime,
            String comment, int newOrder) {
        synchronized (counterLock) {
            transactionCount++;
            fastNewOrderCounter += newOrder;
        }

        if (sessionEndTargetTime != -1 && System.currentTimeMillis() > sessionEndTargetTime) {
            signalTerminalsRequestEnd(true);
        }

        updateStatusLine();
    }

    // #RAIN
    public void allTerminalEnds() {
        sessionEnd = getCurrentTime();
        sessionEndTimestamp = System.currentTimeMillis();
        sessionEndTargetTime = -1;
        printMessage("All terminals finished executing " + sessionEnd);
        endReport();
        terminalsBlockingExit = false;
        printMessage("Session finished!");

        try {
            exportLatencyStatsToCSV("/data/latency_stats.csv"); // #RAIN
        } catch (IOException e) {
            e.printStackTrace();
        }

        // If we opened a per transaction result file, close it.
        if (resultCSV != null) {
            try {
                resultCSV.close();
            } catch (IOException e) {
                log.error(e.getMessage());
            }
        }

        // Stop the OSCollector, if it is active.
        if (osCollector != null) {
            osCollector.stop();
            osCollector = null;
        }

    }

    public jTPCCRandom getRnd() {
        return rnd;
    }

    public void resultAppend(jTPCCTData term) {
        if (resultCSV != null) {
            try {
                resultCSV.write(runID + "," + term.resultLine(sessionStartTimestamp));
            } catch (IOException e) {
                log.error("Term-00, " + e.getMessage());
            }
        }
    }

    private void endReport() {
        long currTimeMillis = System.currentTimeMillis();
        long freeMem = Runtime.getRuntime().freeMemory() / (1024 * 1024);
        long totalMem = Runtime.getRuntime().totalMemory() / (1024 * 1024);
        double tpmC = (6000000 * fastNewOrderCounter / (currTimeMillis - sessionStartTimestamp)) / 100.0;
        double tpmTotal = (6000000 * transactionCount / (currTimeMillis - sessionStartTimestamp)) / 100.0;
        long intervalMillis = sessionEndTimestamp - sessionStartTimestamp; // 计算时间间隔（毫秒）
        double intervalSeconds = (double) intervalMillis / 1000.0; // 转换为秒

        double abortDelay = 20.0;

        if (concurrencyControl.equals("aria") || shouldRetry()) {
            int reNum = getRetryNum();
            intervalSeconds += ((double) reNum * abortDelay / (double) numTerminals / 1000.0);
            addToAbortTime((long) (reNum * abortDelay));
            addToExecTime((long) (reNum * abortDelay));
        }
        // else
        // {
        // int abNum = getAbortNum();
        // intervalSeconds += ((double)abNum*abortDelay/(double)numTerminals/1000.0);
        // addToAbortTime((long)(abNum*abortDelay));
        // addToExecTime((long)(abNum*abortDelay));
        // }

        // add log conflict detection here
        if (needsLogAnalysis()) {
            logConflictAnalyze();
        }

        System.out.println("");
        log.info("Term-00, ");
        log.info("Term-00, ");
        log.info("Term-00, Measured tpmC (NewOrders) = " + tpmC);
        log.info("Term-00, Measured tpmTOTAL = " + tpmTotal);
        log.info("Term-00, Session Start     = " + sessionStart);
        log.info("Term-00, Session End       = " + sessionEnd);
        log.info("Term-00, Session Time(s)   = " + intervalSeconds);

        if (concurrencyControl.equals("mogi")) {
            log.info(
                    "Term-00, Predict Time(s)   = " + (double) ((double) predictTime / (double) numTerminals / 1000.0));
        }

        log.info("Term-00, Lock Time(s)   = " + (double) ((double) lockTime / (double) numTerminals / 1000.0));
        log.info("Term-00, Abort Time(s)   = " + (double) ((double) abortTime / (double) numTerminals / 1000.0));
        log.info("Term-00, Execute Time(s)   = " + (double) ((double) execTime / (double) numTerminals / 1000.0));

        log.info("Term-00, Transaction Count = " + (transactionCount - 1));
        log.info("Term-00, Retry Number\t= " + getRetryNum());
        log.info("Term-00, Aborted Count\t= " + getAbortNum());
        log.info("Term-00, Commit Count\t= " + getCommitNum());
        log.info("Term-00, Waited Count\t= " + getWaitNum());
        log.info("Term-00, Timeout Count\t= " + getTimeoutNum());

        if (needsLogAnalysis()) {
            log.info("Term-00, Error Count\t= " + getErrorNum());
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter("/data/tpcc_output.csv", true))) {
            // 写入列头
            // writer.write(
            // "Protocol,Term,Session Time(s),Predict Time(s),Lock Time(s),Abort
            // Time(s),Execute Time(s),Transaction Count,Retry Number,Aborted Count,Commit
            // Count,Waited Count,Timeout Count,Error Count");
            // writer.newLine();

            // 写入内容行
            writer.write(concurrencyControl + "," + numTerminals + "," + intervalSeconds + "," +
                    (concurrencyControl.equals("mogi") ? (double) predictTime / numTerminals / 1000.0 : 0.0) + "," +
                    (double) lockTime / numTerminals / 1000.0 + "," +
                    (double) abortTime / numTerminals / 1000.0 + "," +
                    (double) execTime / numTerminals / 1000.0 + "," +
                    (transactionCount - 1) + "," +
                    getRetryNum() + "," +
                    getAbortNum() + "," +
                    getCommitNum() + "," +
                    getWaitNum() + "," +
                    getTimeoutNum() + "," +
                    (needsLogAnalysis() ? getErrorNum() : 0));
            writer.newLine();

            System.out.println("Log data has been written to log_output.csv");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void printMessage(String message) {
        log.trace("Term-00, " + message);
    }

    private void errorMessage(String message) {
        log.error("Term-00, " + message);
    }

    private void exit() {
        System.exit(0);
    }

    private String getCurrentTime() {
        return dateFormat.format(new java.util.Date());
    }

    private String getFileNameSuffix() {
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMddHHmmss");
        return dateFormat.format(new java.util.Date());
    }

    synchronized private void updateStatusLine() {
        long currTimeMillis = System.currentTimeMillis();

        if (currTimeMillis > sessionNextTimestamp) {
            StringBuilder informativeText = new StringBuilder("");
            Formatter fmt = new Formatter(informativeText);
            double tpmC = (6000000 * fastNewOrderCounter / (currTimeMillis - sessionStartTimestamp)) / 100.0;
            double tpmTotal = (6000000 * transactionCount / (currTimeMillis - sessionStartTimestamp)) / 100.0;

            sessionNextTimestamp += 1000; /* update this every seconds */

            fmt.format("Term-00, Running Average tpmTOTAL: %.2f", tpmTotal);

            /* XXX What is the meaning of these numbers? */
            recentTpmC = (fastNewOrderCounter - sessionNextKounter) * 12;
            recentTpmTotal = (transactionCount - sessionNextKounter) * 12;
            sessionNextKounter = fastNewOrderCounter;
            fmt.format("    Current tpmTOTAL: %d", recentTpmTotal);

            long freeMem = Runtime.getRuntime().freeMemory() / (1024 * 1024);
            long totalMem = Runtime.getRuntime().totalMemory() / (1024 * 1024);
            fmt.format("    Memory Usage: %dMB / %dMB          ", (totalMem - freeMem), totalMem);

            System.out.print(informativeText);
            for (int count = 0; count < 1 + informativeText.length(); count++)
                System.out.print("\b");
        }
    }
}
