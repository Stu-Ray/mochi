import java.util.Map;
import java.util.HashMap;
import java.util.Objects;

import org.firebirdsql.jdbc.parser.JaybirdSqlParser.nullValue_return;


/* -------------------------- #RAIN -------------------------- */

public class jTPCCDataTS
{
    private final Map<jTPCCDataitem, DataValue> dataMap;

    public jTPCCDataTS() {
        this.dataMap = new HashMap<>();
    }

    // 存储对象
    public boolean containDataKey(jTPCCDataitem key) 
    {
        synchronized (dataMap) {
            return dataMap.containsKey(key);
        }
    }

	public void storeData(jTPCCDataitem key) {
        DataValue value = new DataValue();
        synchronized (dataMap) {
            dataMap.put(key, value);
        }
    }

    public void storeData(jTPCCDataitem key, long rts, long wts) {
        DataValue value = new DataValue(rts, wts);
        synchronized (dataMap) {
            dataMap.put(key, value);  
        }
    }

    // 获取对象
    public DataValue getData(jTPCCDataitem key) {
        DataValue value = new DataValue();
        synchronized (dataMap) {
            value = dataMap.get(key);
        }
        if (value != null) {
            synchronized (value.getLock()) {
                return value;
            }
        }
        return null;
    }

	// 修改RTS和WTS
    public void setRTS(jTPCCDataitem key, long rts) {
        DataValue value = new DataValue();
        synchronized (dataMap) {
            value = dataMap.get(key);
        }
        if (value != null) {
            synchronized (value.getLock()) {
                value.setRTS(rts);
            }
        }
    }

    public void setWTS(jTPCCDataitem key, long wts) {
        DataValue value = new DataValue();
        synchronized (dataMap) {
            value = dataMap.get(key);
        }
        if (value != null) {
            synchronized (value.getLock()) {
                value.setWTS(wts);
            }
        }
    }

    public void addRTS(jTPCCDataitem key, long delta) {
        DataValue value = new DataValue();
        synchronized (dataMap) {
            value = dataMap.get(key);
        }
        if (value != null) {
            synchronized (value.getLock()) {
                value.setRTS(value.getRTS() + delta);
            }
        }
    }

    public void addWTS(jTPCCDataitem key, long delta) {
        DataValue value = new DataValue();
        synchronized (dataMap) {
            value = dataMap.get(key);
        }
        if (value != null) {
            synchronized (value.getLock()) {
                value.setWTS(value.getWTS() + delta);
            }
        }
    }

    public static class DataValue {
        private long    rts;
        private long    wts;
        private Object  lock;

        public DataValue() {
            this.rts = 0;
            this.wts = 0;
            this.lock = new Object();
        }

        public DataValue(long rts, long wts) {
            this.rts = rts;
            this.wts = wts;
            this.lock = new Object();
        }

        // Getters and setters for timestamps
        public long getRTS() { return rts; }
        public void setRTS(long rts) { this.rts = rts; }

        public long getWTS() { return wts; }
        public void setWTS(long wts) { this.wts = wts; }

        public Object getLock() 
        { 
            if(lock == null)
                lock = new Object();
            return lock; 
        }
    }
}

/* -------------------------- #RAIN -------------------------- */