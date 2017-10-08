package net.huutonauru.neural;

import spock.lang.Specification;

class NeuronSpec extends Specification {

    void "create class test"() {
        when:
        Neuron neuron = new Neuron()

        then:
        neuron
    }
}
