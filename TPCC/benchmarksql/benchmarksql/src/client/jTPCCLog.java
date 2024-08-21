public class jTPCCLog {
    private int ooid;

    private jTPCCDataitem dataitem;

    public jTPCCLog() {
        this.ooid = -1;
        this.dataitem = new jTPCCDataitem();
    }

    public jTPCCLog(int ooid, jTPCCDataitem dataitem) {
        this.ooid = ooid;
        this.dataitem = dataitem;
    }

    public int getOoid() {
        return ooid;
    }

    public jTPCCDataitem getDataitem() {
        return dataitem;
    }

    public void setOoid(int ooid) {
        this.ooid = ooid;
    }

    public void setDataitem(jTPCCDataitem dataitem) {
        this.dataitem = dataitem;
    }

    public boolean isRead()
    {
        return (dataitem.getTypeId() == 1);
    }

    public boolean isWrite()
    {
        return (dataitem.getTypeId() == 2 || dataitem.getTypeId() == 3 || dataitem.getTypeId() == 4); // UPDATE\DELETE\INSERT
    }

    @Override
    public int hashCode() {
        int result = ((dataitem == null) ? 0 : dataitem.hashCode());
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
        jTPCCLog other = (jTPCCLog) obj;
        if (dataitem == null) {
            if (other.dataitem != null)
                return false;
        } 
        else if (this.getOoid() != other.getOoid())
            return false;
        else if (!dataitem.equals(other.dataitem))
            return false;
        return true;
    }

}
