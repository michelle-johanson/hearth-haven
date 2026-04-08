import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard, Users, Home, Heart, Globe, Share2, FileText,
  X, PanelLeftClose, PanelLeftOpen,
} from 'lucide-react';

const navItems = [
  { to: '/admin', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/cases', label: 'Cases', icon: Users },
  { to: '/safehouse-management', label: 'Safehouses', icon: Home },
  { to: '/donors', label: 'Donors', icon: Heart },
  { to: '/outreach', label: 'Outreach', icon: Globe },
  { to: '/social-media', label: 'Social Media', icon: Share2 },
  { to: '/reports', label: 'Reports', icon: FileText },
];

type AdminSidebarProps = {
  open: boolean;
  onClose: () => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
};

function AdminSidebar({ open, onClose, collapsed, onToggleCollapse }: AdminSidebarProps) {
  const { pathname } = useLocation();

  const isActive = (to: string) =>
    to === '/admin' ? pathname === '/admin' : pathname.startsWith(to);

  return (
    <>
      {/* Mobile backdrop */}
      {open && (
        <div
          className="fixed inset-x-0 top-[57px] bottom-0 z-20 bg-black/30 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed top-[57px] bottom-0 left-0 z-30 flex shrink-0 flex-col border-r border-gray-200 bg-white transition-all duration-200 dark:border-gray-700 dark:bg-gray-900 lg:static lg:top-0 lg:translate-x-0 ${
          collapsed ? 'w-16' : 'w-60'
        } ${open ? 'translate-x-0' : '-translate-x-full'}`}
      >
        {/* Header with collapse toggle */}
        <div className={`flex items-center pt-3 pb-2 lg:pt-6 ${collapsed ? 'justify-center px-2' : 'justify-between px-5'}`}>
          {!collapsed && (
            <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
              Admin
            </h2>
          )}
          {/* Mobile close button */}
          <button
            className="btn-icon lg:hidden"
            onClick={onClose}
            aria-label="Close sidebar"
          >
            <X className="h-4 w-4" />
          </button>
          {/* Desktop collapse toggle */}
          <button
            onClick={onToggleCollapse}
            className="btn-icon hidden lg:inline-flex"
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
          </button>
        </div>

        <nav className={`flex-1 py-2 ${collapsed ? 'px-2' : 'px-5'}`}>
          <ul className="space-y-1">
            {navItems.map(({ to, label, icon: Icon }) => (
              <li key={to}>
                <Link
                  to={to}
                  onClick={onClose}
                  title={collapsed ? label : undefined}
                  className={`flex w-full min-w-max items-center gap-3 whitespace-nowrap rounded-lg py-2 text-sm font-medium no-underline transition ${
                    collapsed ? 'justify-center px-2' : 'px-3 text-left'
                  } ${
                    isActive(to)
                      ? 'bg-orange-50 text-orange-700 dark:bg-orange-500/10 dark:text-orange-400'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-white'
                  }`}
                >
                  <Icon className="h-4 w-4 shrink-0" />
                  {!collapsed && label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </aside>
    </>
  );
}

export default AdminSidebar;
