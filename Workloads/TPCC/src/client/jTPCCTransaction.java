import java.util.ArrayList;
import java.util.List;

public class jTPCCTransaction {
    private int ooid;
    private long inputHash;
    private int inputSize;
    private long outputHash;
    private int outputSize;
    private List<jTPCCDataitem> list;

    public jTPCCTransaction() {
        this.ooid = -1;
        this.inputHash = 0;
        this.inputSize = 0;
        this.outputHash = 0;
        this.outputSize = 0;
        this.list = new ArrayList<>();
    }

    public jTPCCTransaction(int ooid) {
        this.ooid = ooid;
        this.inputHash = 0;
        this.inputSize = 0;
        this.outputHash = 0;
        this.outputSize = 0;
        this.list = new ArrayList<>();
    }

    public jTPCCTransaction(int ooid, long inputHash, int inputSize, long outputHash, int outputSize,
            List<jTPCCDataitem> list) {
        this.ooid = ooid;
        this.inputHash = inputHash;
        this.inputSize = inputSize;
        this.outputHash = outputHash;
        this.outputSize = outputSize;
        this.list = list;
    }

    public void addDataToList(jTPCCDataitem dataitem, boolean input_bool) {
        this.list.add(dataitem);

        if (input_bool)
            this.inputHash += dataitem.getHashValue();
        else
            this.outputHash += dataitem.getHashValue();
    }

    public int getOoid() {
        return ooid;
    }

    public long getInputHash() {
        return inputHash;
    }

    public int getInputSize() {
        return inputSize;
    }

    public long getOutputHash() {
        return outputHash;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public List<jTPCCDataitem> getList() {
        return list;
    }

    public void setOoid(int ooid) {
        this.ooid = ooid;
    }

    public void setInputHash(long inputHash) {
        this.inputHash = inputHash;
    }

    public void setInputSize(int inputSize) {
        this.inputSize = inputSize;
    }

    public void setOutputHash(long outputHash) {
        this.outputHash = outputHash;
    }

    public void setOutputSize(int outputSize) {
        this.outputSize = outputSize;
    }

    public void setList(List<jTPCCDataitem> list) {
        this.list = list;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        jTPCCTransaction other = (jTPCCTransaction) obj;
        if (ooid != other.ooid)
            return false;
        if (inputHash != other.inputHash)
            return false;
        if (inputSize != other.inputSize)
            return false;
        if (outputHash != other.outputHash)
            return false;
        if (outputSize != other.outputSize)
            return false;
        if (list == null) {
            if (other.list != null)
                return false;
        } else if (!list.equals(other.list))
            return false;
        return true;
    }

}
