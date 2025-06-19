public class jTPCCDataitem 
{
    private int typeId;
    private int tableId;
    private int wid;
    private int did;
    private int cid;
    private int iid;

    public jTPCCDataitem() {
        this.typeId = 0;
        this.tableId = 0;
        this.wid = 0;
        this.did = 0;
        this.cid = 0;
        this.iid = 0;
    }
    
    public jTPCCDataitem(int typeId, int tableId, int wid, int did, int cid, int iid) 
    {
        this.typeId = typeId;
        this.tableId = tableId;
        this.wid = wid;
        this.did = did;
        this.cid = cid;
        this.iid = iid;
    }

    public int getTypeId() {
        return typeId;
    }
    public int getTableId() {
        return tableId;
    }
    public int getWid() {
        return wid;
    }
    public int getDid() {
        return did;
    }
    public int getCid() {
        return cid;
    }
    public int getIid() {
        return iid;
    }
    
    public void setTypeId(int typeId) {
        this.typeId = typeId;
    }
    public void setTableId(int tableId) {
        this.tableId = tableId;
    }
    public void setWid(int wid) {
        this.wid = wid;
    }
    public void setDid(int did) {
        this.did = did;
    }
    public void setCid(int cid) {
        this.cid = cid;
    }
    public void setIid(int iid) {
        this.iid = iid;
    }

    public long getHashValue() {
        final long prime = 13;
        long result = 7;
        result = prime * result + tableId;
        result = prime * result + wid;
        result = prime * result + did;
        result = prime * result + cid;
        result = prime * result + iid;
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        jTPCCDataitem other = (jTPCCDataitem) obj;
        if (tableId != other.tableId)
            return false;
        if (wid != other.wid)
            return false;
        if (did != other.did)
            return false;
        if (cid != other.cid)
            return false;
        if (iid != other.iid)
            return false;
        return true;
    }

    @Override
    public int hashCode() {
        final int prime = 13;
        int result = 7;
        result = prime * result + tableId;
        result = prime * result + wid;
        result = prime * result + did;
        result = prime * result + cid;
        result = prime * result + iid;
        return result;
    }
    
}
