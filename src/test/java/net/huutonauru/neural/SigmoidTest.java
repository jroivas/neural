package net.huutonauru.neural;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

public class SigmoidTest {
    @Test
    public void DefaultSigmoidTranfer() {
        Sigmoid s = new DefaultSigmoid();
        assertEquals(s.transfer(1), 1);
    }

    @Test
    public void LogSigmoidTranfer() {
        Sigmoid s = new LogSigmoid(0, 1);
        assertNotEquals(s.transfer(1), 1);
    }
}
