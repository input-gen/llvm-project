import interactive_host
import json
import sys


def main(args):
    class Advisor:
        to_return = False
        counter = 0

        def advice(self, _):
            return int(args[0])

        def before_advice(self, tc, fc):
            print("BEFORE_ADVICE")
            json_str = fc.readline()
            print(json_str)
            assert 'heuristic' in json.loads(json_str)
            heuristic = int.from_bytes(fc.read(8), byteorder=sys.byteorder, signed=True)
            print(heuristic)
            fc.readline()

        def after_advice(self, tc, fc):
            print("AFTER_ADVICE")
            json_str = fc.readline()
            print(json_str)
            assert 'action' in json.loads(json_str)
            action = bool(int.from_bytes(fc.read(1)))
            print(action)
            fc.readline()
            if args[1] == 'instrument':
                tc.write(bytes([1]))
                begin = ("test_loop_begin_" + str(self.counter)).encode('ascii') + bytes([0])
                end = ("test_loop_end_" + str(self.counter)).encode('ascii') + bytes([0])
                self.counter += 1
                tc.write(begin)
                tc.write(end)
                tc.flush()
            else:
                # Respond that we do not want instrumentation
                tc.write(bytes([0]))
                tc.flush()

    a = Advisor()
    interactive_host.run_interactive(args[2],
                                     a.advice,
                                     args[3:],
                                     a.before_advice,
                                     a.after_advice)


if __name__ == "__main__":
    main(sys.argv[1:])
