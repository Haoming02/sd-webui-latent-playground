from modules import scripts


def grid_reference():
    for data in scripts.scripts_data:
        if data.script_class.__module__ == "xyz_grid.py" and hasattr(data, "module"):
            return data.module

    raise SystemError("Could not find X/Y/Z Plot...")


def xyz_support(cache: dict):

    def apply_field(field):
        def _(p, x, xs):
            cache.update({field: x})

        return _

    xyz_grid = grid_reference()

    extra_axis_options = [
        xyz_grid.AxisOption(
            "[Playground] Channel",
            int,
            apply_field("chn"),
        ),
        xyz_grid.AxisOption(
            "[Playground] Delta",
            float,
            apply_field("delta"),
        ),
    ]

    xyz_grid.axis_options.extend(extra_axis_options)
