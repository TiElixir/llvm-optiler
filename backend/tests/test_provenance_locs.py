from backend.provenance.locs import lines_of, parse_loc


def test_file_line_col():
    assert lines_of('loc("source/model.py":7:0)') == [7]
    assert lines_of('loc("/run/dir/source/model.py":7:12)') == [7]


def test_unknown_and_foreign_files_have_no_user_line():
    assert lines_of("loc(unknown)") == []
    assert lines_of('loc("/site-packages/torch/nn/modules/linear.py":134:0)') == []
    assert lines_of("garbage((") == []


def test_callsite_resolves_to_the_caller_frame():
    t = 'loc(callsite("/x/site-packages/torch/nn/modules/linear.py":134:0 at "source/model.py":12:0))'
    assert lines_of(t) == [12]
    nested = 'loc(callsite("a.py":1:0 at callsite("b.py":2:0 at "source/model.py":30:0)))'
    assert lines_of(nested) == [30]
    assert lines_of('loc(callsite("source/model.py":5:0 at "other.py":9:0))') == [5]  # callee fallback


def test_fused_locations_link_every_contributing_line():
    t = 'loc(fused[loc("source/model.py":7:0), "source/model.py":8:0, callsite("l.py":1:0 at "source/model.py":9:0)])'
    assert lines_of(t.replace('loc("source/model.py":7:0)', '"source/model.py":7:0')) == [7, 8, 9]
    assert lines_of('loc(fused<"tag">["source/model.py":3:0, unknown])') == [3]


def test_name_locations_unwrap_their_child():
    assert lines_of('loc("fx_node_name"("source/model.py":4:0))') == [4]
    assert lines_of('loc("just a name")') == []


def test_range_locations_and_parse_shape():
    assert lines_of('loc("source/model.py":4:2 to :9)') == [4]
    assert parse_loc("loc(unknown)") == ("unknown",)
