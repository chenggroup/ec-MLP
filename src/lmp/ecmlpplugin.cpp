// SPDX-License-Identifier: LGPL-3.0-or-later
#include "comm.h"
#include "command.h"
#include "error.h"
#include "lammpsplugin.h"
#include "update.h"
#include "utils.h"
#include "version.h"

#include "verlet_split_kspace.h" // Include the declaration of VerletSplitKSpace
#include "verlet_split_dplr.h"   // Include the declaration of VerletSplitDPLR

namespace LAMMPS_NS
{

  // Define the creator function
  static Integrate *VerletSplitDplr_creator(LAMMPS *lmp, int narg, char **arg)
  {
    return new VerletSplitDPLR(lmp, narg, arg);
  }
  static Integrate *VerletSplitKSpace_creator(LAMMPS *lmp, int narg, char **arg)
  {
    return new VerletSplitKSpace(lmp, narg, arg);
  }

  // Define the command class IntegrateRegisterCommand to register VerletSplitDPLR
  // into integrate_map
  class IntegrateRegisterCommand : public Command
  {
  public:
    IntegrateRegisterCommand(LAMMPS *lmp) : Command(lmp) {}

    void command(int narg, char **arg) override
    {
      if (narg < 1)
      {
        error->all(
            FLERR,
            "Illegal add_run_style command. Usage: add_run_style [integrator1_name] [integrator2_name] ...");
        return;
      }

      std::string integrate_name = "";

      for (int i = 0; i < narg; i++)
      {
        if (strcmp(arg[i], "verlet/split/dplr") == 0)
        {
          integrate_name = "verlet/split/dplr"; // Name of the integrate style
          if (lmp->update->integrate_map->find(integrate_name) ==
              lmp->update->integrate_map->end())
          {
            (*lmp->update->integrate_map)[integrate_name] =
                (Update::IntegrateCreator)(lammpsplugin_factory2 *)&VerletSplitDplr_creator;
          }
          else
          {
            error->all(FLERR,
                       "Integrate style %s already exists in integrate_map",
                       integrate_name.c_str());
          }
        }
        else if (strcmp(arg[i], "verlet/split/kspace") == 0)
        {
          integrate_name = "verlet/split/kspace"; // Name of the integrate style
          if (lmp->update->integrate_map->find(integrate_name) ==
              lmp->update->integrate_map->end())
          {
            (*lmp->update->integrate_map)[integrate_name] =
                (Update::IntegrateCreator)(lammpsplugin_factory2 *)&VerletSplitKSpace_creator;
          }
          else
          {
            error->all(FLERR,
                       "Integrate style %s already exists in integrate_map",
                       integrate_name.c_str());
          }
        }
        else
        {
          error->all(FLERR, "Unsupported run_style: {}", arg[i]);
        }

        if (lmp->comm->me == 0)
        {
          utils::logmesg(lmp, "Added run_style {} successfully.\n",
                         integrate_name.c_str());
        }
      }
    }
  };

  // Define the factory function for IntegrateRegisterCommand
  static Command *integrate_register_command_creator(LAMMPS *lmp)
  {
    return new IntegrateRegisterCommand(lmp);
  }

} // namespace LAMMPS_NS

// Plugin initialization function to register IntegrateRegisterCommand as a new
// LAMMPS command
extern "C" void lammpsplugin_init(void *lmp_ptr, void *handle, void *regfunc)
{
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc)regfunc;
  lammpsplugin_t plugin;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "command";
  plugin.name = "add_run_style";
  plugin.info = "Command to register custom time integrator";
  plugin.author = "Si-Yuan Han";
  plugin.creator.v1 =
      (lammpsplugin_factory1 *)&LAMMPS_NS::integrate_register_command_creator;
  plugin.handle = handle;

  (*register_plugin)(&plugin, lmp_ptr);
}
