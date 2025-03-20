// SPDX-License-Identifier: LGPL-3.0-or-later
#include "comm.h"
#include "command.h"
#include "error.h"
#include "lammpsplugin.h"
#include "update.h"
#include "utils.h"
#include "verlet_split_dplr.h" // Include the declaration of VerletSplitDplr
#include "version.h"

#include "pppm_dplr.h"

namespace LAMMPS_NS {

// Define the creator function for VerletSplitDplr
static Integrate *VerletSplitDplr_creator(LAMMPS *lmp, int narg, char **arg) {
  return new VerletSplitDplr(lmp, narg, arg);
}

// Define the command class IntegrateRegisterCommand to register VerletSplitDplr
// into integrate_map
class IntegrateRegisterCommand : public Command {
public:
  IntegrateRegisterCommand(LAMMPS *lmp) : Command(lmp) {}

  void command(int narg, char **arg) override {
    if (narg != 1) {
      error->all(
          FLERR,
          "Illegal register_integrate command. Usage: register_integrate dplr");
      return;
    }

    std::string arg_value =
        arg[0]; // Accessing the first argument after the command name

    if (arg_value == "dplr") {
      // std::string integrate_name = "verlet/split/dplr";  // Name of the
      // integrate style
      std::string integrate_name =
          "verlet/split/dplr"; // Name of the integrate style
      // Register the creator function into integrate_map
      if (lmp->update->integrate_map->find(integrate_name) ==
          lmp->update->integrate_map->end()) {
        // lammpsplugin_factory2* factory = (lammpsplugin_factory2*)
        // &VerletSplitDplr_creator;
        //(*lmp->update->integrate_map)[integrate_name] =
        //(Update::IntegrateCreator)factory;

        (*lmp->update->integrate_map)[integrate_name] =
            (Update::IntegrateCreator)(
                lammpsplugin_factory2 *)&VerletSplitDplr_creator;

        if (lmp->comm->me == 0) {
          utils::logmesg(lmp, "Registered integrate style successfully.\n",
                         integrate_name.c_str());
        }
      } else {
        error->all(FLERR,
                   "Integrate style '%s' already exists in integrate_map",
                   integrate_name.c_str());
      }
    } else {
      error->all(FLERR, "Illegal argument for register_integrate command. "
                        "Usage: register_integrate dplr");
    }
  }
};

// Define the factory function for IntegrateRegisterCommand
static Command *integrate_register_command_creator(LAMMPS *lmp) {
  return new IntegrateRegisterCommand(lmp);
}

} // namespace LAMMPS_NS

// Plugin initialization function to register IntegrateRegisterCommand as a new
// LAMMPS command
extern "C" void lammpsplugin_init(void *lmp_ptr, void *handle, void *regfunc) {
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc)regfunc;
  lammpsplugin_t plugin;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "command";
  plugin.name = "register_integrate";
  plugin.info = "Command to register custom integrate style VerletSplitDplr";
  plugin.author = "Your Name";
  plugin.creator.v1 =
      (lammpsplugin_factory1 *)&LAMMPS_NS::integrate_register_command_creator;
  plugin.handle = handle;

  (*register_plugin)(&plugin, lmp_ptr);
}
